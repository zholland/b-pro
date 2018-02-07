/****************************************************************************************
 ** Implementation of Sarsa(lambda). It implements Fig. 8.8 (Linear, gradient-descent
 ** Sarsa(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An
 ** Introduction. 1st edition. 1988."
 ** Some updates are made to make it more efficient, as not iterating over all features.
 **
 ** TODO: Make it as efficient as possible.
 **
 ** Author: Marlos C. Machado
 ***************************************************************************************/
#ifndef MATHEMATICS_H
#define MATHEMATICS_H
#include "../../../common/Mathematics.hpp"
#endif

#ifndef TIMER_H
#define TIMER_H

#include "../../../common/Timer.hpp"

#endif

#include "SarsaLearner.hpp"
#include <stdio.h>
#include <math.h>
#include <set>

#include "../../../lodepng.hpp"
#include <iostream>

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/kernels/no_op.h>

using namespace tensorflow;
using namespace std;

typedef unsigned char pixel_t;

//using google::dense_hash_map;

SarsaLearner::SarsaLearner(ALEInterface& ale, BlobTimeFeatures *features, Parameters *param,int seed) : RLLearner(ale, param,seed) {
    
    totalNumberFrames = 0.0;
    maxFeatVectorNorm = 1;
    maxPlanFeatVectorNorm = 1;
    saveThreshold = 0;

    delta = 0.0;
    alpha = param->getAlpha();
    learningRate = alpha;
    lambda = param->getLambda();
    numGroups = 0;
    traceThreshold = param->getTraceThreshold();
    numFeatures = features->getNumberOfFeatures();
    toSaveCheckPoint = param->getToSaveCheckPoint();
    saveWeightsEveryXFrames = param->getFrequencySavingWeights();
    pathWeightsFileToLoad = param->getPathToWeightsFiles();
    randomNoOp = param->getRandomNoOp();
    noOpMax = param->getNoOpMax();
    numStepsPerAction = param->getNumStepsPerAction();

    backgroundPath = param->getBackgroundPath();
    palettePath = param->getPalettePath();
    modelPath = param->getLearnedModelPath();

    planningSteps = param->getPlanningSteps();
    planningIterations = param->getPlanningIterations();
    planBufferSize = param->getPlanBufferSize();

    for (int i = 0; i < numActions; i++) {
        //Initialize Q;
        Q.push_back(0);
        Qplan.push_back(0);
        Qnext.push_back(0);
        //Initialize e:
        e.push_back(vector<float>());
        ePlan.push_back(vector<float>());
        w.push_back(vector<float>());
        oldWeights.push_back(vector<float>());
        nonZeroElig.push_back(vector<long long>());
        planNonZeroElig.push_back(vector<long long>());
    }
    episodePassed = 0;
    featureTranslate.clear();
    featureTranslate.max_load_factor(0.5);
    if (toSaveCheckPoint) {
        checkPointName = param->getCheckPointName();
        //load CheckPoint
        ifstream checkPointToLoad;
        string checkPointLoadName = checkPointName + "-checkPoint.txt";
        checkPointToLoad.open(checkPointLoadName.c_str());
        if (checkPointToLoad.is_open()) {
            loadCheckPoint(checkPointToLoad);
//            remove(checkPointLoadName.c_str());
        }
        saveThreshold = (totalNumberFrames / saveWeightsEveryXFrames) * saveWeightsEveryXFrames;
        ofstream learningConditionFile;
        nameForLearningCondition = checkPointName+"-learningCondition-Frames"+to_string(static_cast<long long>(saveThreshold))+"-finished.txt";
        string previousNameForLearningCondition =checkPointName +"-learningCondition.txt";
        rename(previousNameForLearningCondition.c_str(), nameForLearningCondition.c_str());
        saveThreshold += saveWeightsEveryXFrames;
        learningConditionFile.open(nameForLearningCondition, ios_base::app);
        learningConditionFile.close();
    }
}

SarsaLearner::~SarsaLearner() {}

void SarsaLearner::updateQValuesWithWeights(vector<long long>& BlobTimeFeatures, vector<float>& QValues, vector<vector<float>>& weights) {
    unsigned long long featureSize = BlobTimeFeatures.size();
    for (int a = 0; a < numActions; ++a) {
        float sumW = 0;
        for (unsigned long long i = 0; i < featureSize; ++i) {
            sumW = sumW + weights[a][BlobTimeFeatures[i]] * groups[BlobTimeFeatures[i]].numFeatures;
        }
        QValues[a] = sumW;
    }
}

void SarsaLearner::updateQValues(vector<long long> &BlobTimeFeatures, vector<float> &QValues) {
    unsigned long long featureSize = BlobTimeFeatures.size();
    for (int a = 0; a < numActions; ++a) {
        float sumW = 0;
        for (unsigned long long i = 0; i < featureSize; ++i) {
            sumW = sumW + w[a][BlobTimeFeatures[i]] * groups[BlobTimeFeatures[i]].numFeatures;
        }
        QValues[a] = sumW;
    }
}

void SarsaLearner::updateReplTrace(int action, vector<long long> &BlobTimeFeatures) {
    //e <- gamma * lambda * e
    for (unsigned int a = 0; a < nonZeroElig.size(); a++) {
        long long numNonZero = 0;
        for (unsigned long long i = 0; i < nonZeroElig[a].size(); i++) {
            long long idx = nonZeroElig[a][i];
            //To keep the trace sparse, if it is
            //less than a threshold it is zero-ed.
            e[a][idx] = gamma * lambda * e[a][idx];
            if (e[a][idx] < traceThreshold) {
                e[a][idx] = 0;
            } else {
                nonZeroElig[a][numNonZero] = idx;
                numNonZero++;
            }
        }
        nonZeroElig[a].resize(numNonZero);
    }

    //For all i in Fa:
    for (unsigned int i = 0; i < F.size(); i++) {
        long long idx = BlobTimeFeatures[i];
        //If the trace is zero it is not in the vector
        //of non-zeros, thus it needs to be added
        if (e[action][idx] == 0) {
            nonZeroElig[action].push_back(idx);
        }
        e[action][idx] = 1;
    }
}

void SarsaLearner::updatePlanReplTrace(int action, vector<long long> &BlobTimeFeatures) {
    //e <- gamma * lambda * e
    for (unsigned int a = 0; a < planNonZeroElig.size(); a++) {
        long long numNonZero = 0;
        for (unsigned long long i = 0; i < planNonZeroElig[a].size(); i++) {
            long long idx = planNonZeroElig[a][i];
            //To keep the trace sparse, if it is
            //less than a threshold it is zero-ed.
            ePlan[a][idx] = gamma * lambda * ePlan[a][idx];
            if (ePlan[a][idx] < traceThreshold) {
                ePlan[a][idx] = 0;
            } else {
                planNonZeroElig[a][numNonZero] = idx;
                numNonZero++;
            }
        }
        planNonZeroElig[a].resize(numNonZero);
    }

    //For all i in Fa:
    for (unsigned int i = 0; i < Fplan.size(); i++) {
        long long idx = BlobTimeFeatures[i];
        //If the trace is zero it is not in the vector
        //of non-zeros, thus it needs to be added
        if (ePlan[action][idx] == 0) {
            planNonZeroElig[action].push_back(idx);
        }
        ePlan[action][idx] = 1;
    }
}

void SarsaLearner::updateAcumTrace(int action, vector<long long> &BlobTimeFeatures) {
    //e <- gamma * lambda * e
    for (unsigned int a = 0; a < nonZeroElig.size(); a++) {
        long long numNonZero = 0;
        for (unsigned int i = 0; i < nonZeroElig[a].size(); i++) {
            long long idx = nonZeroElig[a][i];
            //To keep the trace sparse, if it is
            //less than a threshold it is zero-ed.
            e[a][idx] = gamma * lambda * e[a][idx];
            if (e[a][idx] < traceThreshold) {
                e[a][idx] = 0;
            } else {
                nonZeroElig[a][numNonZero] = idx;
                numNonZero++;
            }
        }
        nonZeroElig[a].resize(numNonZero);
    }

    //For all i in Fa:
    for (unsigned int i = 0; i < F.size(); i++) {
        long long idx = BlobTimeFeatures[i];
        //If the trace is zero it is not in the vector
        //of non-zeros, thus it needs to be added
        if (e[action][idx] == 0) {
            nonZeroElig[action].push_back(idx);
        }
        e[action][idx] += 1;
    }
}

void SarsaLearner::sanityCheck() {
    for (int i = 0; i < numActions; i++) {
        if (fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/) {
            printf("It seems your algorithm diverged!\n");
            exit(0);
        }
    }
}

//To do: we do not want to save weights that are zero
void SarsaLearner::saveCheckPoint(int episode, int totalNumberFrames, vector<float> &episodeResults, int &frequency,
                                  vector<int> &episodeFrames, vector<double> &episodeFps) {
    ofstream learningConditionFile;
    string newNameForLearningCondition =
            checkPointName + "-learningCondition-Frames" + to_string(static_cast<long long>(saveThreshold)) +
            "-writing.txt";
    int renameReturnCode = rename(nameForLearningCondition.c_str(), newNameForLearningCondition.c_str());
    if (renameReturnCode == 0) {
        nameForLearningCondition = newNameForLearningCondition;
        learningConditionFile.open(nameForLearningCondition.c_str(), ios_base::app);
        int numEpisode = episodeResults.size();
        for (int index = 0; index < numEpisode; index++) {
            learningConditionFile << "Episode " << episode - numEpisode + 1 + index << ": " << episodeResults[index]
                                  << " points,  " << episodeFrames[index] << " frames,  " << episodeFps[index]
                                  << " fps." << endl;
        }
        episodeResults.clear();
        episodeFrames.clear();
        episodeFps.clear();
        learningConditionFile.close();
        newNameForLearningCondition.replace(newNameForLearningCondition.end() - 11,
                                            newNameForLearningCondition.end() - 4, "finished");
        rename(nameForLearningCondition.c_str(), newNameForLearningCondition.c_str());
        nameForLearningCondition = newNameForLearningCondition;
    }

    //write parameters checkPoint
    string currentCheckPointName = checkPointName+"-checkPoint-Frames"+to_string(static_cast<long long>(saveThreshold))+"-writing.txt";
    ofstream checkPointFile;
    checkPointFile.open(currentCheckPointName.c_str());
    checkPointFile << (*agentRand) << endl;
    checkPointFile << totalNumberFrames << endl;
    checkPointFile << episode << endl;
    checkPointFile << firstReward << endl;
    checkPointFile << maxFeatVectorNorm << endl;
    checkPointFile << numGroups << endl;
    checkPointFile << featureTranslate.size() << endl;
    vector<int> nonZeroWeights;
    for (unsigned long long groupIndex = 0; groupIndex < numGroups; ++groupIndex) {
        nonZeroWeights.clear();
        for (unsigned long long a = 0; a < w.size(); a++) {
            if (w[a][groupIndex] != 0) {
                nonZeroWeights.push_back(a);
            }
        }
        checkPointFile << nonZeroWeights.size();
        for (int i = 0; i < nonZeroWeights.size(); ++i) {
            int action = nonZeroWeights[i];
            checkPointFile << " " << action << " " << w[action][groupIndex];
        }
        checkPointFile << "\t";
    }
    checkPointFile << endl;

    for (auto it = featureTranslate.begin(); it != featureTranslate.end(); ++it) {
        checkPointFile << it->first << " " << it->second << "\t";
    }
    checkPointFile << endl;
    checkPointFile.close();
    
    string previousVersionCheckPoint = checkPointName+"-checkPoint-Frames"+to_string(static_cast<long long>(saveThreshold-saveWeightsEveryXFrames))+"-finished.txt";
    if((saveThreshold-saveWeightsEveryXFrames)%50000000 != 0){
        remove(previousVersionCheckPoint.c_str());
    }
    string oldCheckPointName = currentCheckPointName;
    currentCheckPointName.replace(currentCheckPointName.end() - 11, currentCheckPointName.end() - 4, "finished");
    rename(oldCheckPointName.c_str(), currentCheckPointName.c_str());

}

void SarsaLearner::loadCheckPoint(ifstream &checkPointToLoad) {
    checkPointToLoad >> (*agentRand);
    cout << (*agentRand) << "\n";
    checkPointToLoad >> totalNumberFrames;
    cout << totalNumberFrames << "\n";
    while (totalNumberFrames < 1000) {
        checkPointToLoad >> totalNumberFrames;
    }
    cout << totalNumberFrames << "\n";
    checkPointToLoad >> episodePassed;
    checkPointToLoad >> firstReward;
    checkPointToLoad >> maxFeatVectorNorm;
    learningRate = alpha / float(maxFeatVectorNorm);
    checkPointToLoad >> numGroups;
    long long numberOfFeaturesSeen;
    checkPointToLoad >> numberOfFeaturesSeen;
    for (unsigned long long index = 0; index < numGroups; ++index) {
        Group agroup;
        agroup.numFeatures = 0;
        agroup.features.clear();
        groups.push_back(agroup);
    }
    for (unsigned a = 0; a < w.size(); a++) {
        w[a].resize(numGroups, 0.00);
        e[a].resize(numGroups, 0.00);
        ePlan[a].resize(numGroups, 0.00);
    }
    int action;
    float weight;
    int numNonZeroWeights;
    for (unsigned long long groupIndex = 0; groupIndex < numGroups; ++groupIndex) {
        checkPointToLoad >> numNonZeroWeights;
        for (unsigned int i = 0; i < numNonZeroWeights; ++i) {
            checkPointToLoad >> action;
            checkPointToLoad >> weight;
            w[action][groupIndex] = weight;
        }
    }

    long long featureIndex;
    long long featureToGroup;
    while (checkPointToLoad >> featureIndex && checkPointToLoad >> featureToGroup) {
        featureTranslate[featureIndex] = featureToGroup;
        groups[featureToGroup - 1].numFeatures += 1;
    }
    checkPointToLoad.close();
}

void SarsaLearner::oneHot(Tensor & tensor, int tensorSize, int index) {
    for (int i = 0; i < tensorSize; i++) {
        if (i == index) {
            tensor.tensor<float, 3>()(0, 0, i) = 1.0;
        } else {
            tensor.tensor<float, 3>()(0, 0, i) = 0.0;
        }
    }
}

void SarsaLearner::learnPolicy(ALEInterface &ale, BlobTimeFeatures *features) {
    std::string actionFileName = checkPointName + "-actions.txt";
    std::ofstream actionFile;
    actionFile.open(actionFileName.c_str());

    // Load background
    ifstream inFile;
    inFile.open(backgroundPath.c_str());
    if (!inFile) {
        std::cerr << "Unable to open file: "<<backgroundPath;
        exit(1);   // call system to stop
    }
    vector<float> backgroundChannelMeans(3);
    float value;
    int channel = 0;
    while (inFile >> value) {
        backgroundChannelMeans[channel] = value;
        channel+=1;
    }
    inFile.close();
    // Load palette
    inFile.open(palettePath.c_str());
    if (!inFile) {
        std::cerr << "Unable to open file: "<<palettePath;
        exit(1);   // call system to stop
    }
    vector<int> palette;
    int paletteIndex;
    while (inFile >> paletteIndex) {
        palette.push_back(paletteIndex);
    }
    inFile.close();

    struct timeval tvBegin, tvEnd, tvDiff;
    vector<float> reward;
    double elapsedTime;
    double cumReward = 0, prevCumReward = 0;
    sawFirstReward = 0;
    firstReward = 1.0;
    vector<float> episodeResults;
    vector<int> episodeFrames;
    vector<double> episodeFps;

    //std::random_device rd;  //Will be used to obtain a seed for the random number engine
    //std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    //std::uniform_real_distribution<double> dis(0.0, 1.0);

    vector<float> oldWeightsQ(numActions);

    int bufferIndex = 0;
    int maxBufferSize = 10000;
    int actualBufferSize = planBufferSize;
    if (planBufferSize < 0) {
        actualBufferSize = maxBufferSize;
    }

    vector<ALEState> stateBuffer(actualBufferSize);
    vector<ALEScreen> screenBuffer(actualBufferSize, ale.getScreen());
    vector<float> rewardBuffer(actualBufferSize);
    vector<vector<vector<tuple<int,int>>>> prevBlobsBuffer(actualBufferSize);
    vector<vector<int>> prevBlobsActiveColorsBuffer(actualBufferSize);

    vector<vector<tuple<int,int>>> prevBlobsSave;
    vector<int> prevBlobsActiveColorsSave;

    long long trueFeatureSize = 0;
    long long truePlanFeatureSize = 0;
    long long trueFnextSize = 0;
    long long truePlanFnextSize = 0;

    int stepCount = 0;

    const string pathToGraph = modelPath + "/checkpoint_inference.meta";
    const string checkpointPath = modelPath + "/checkpoint_inference";
    // Init tensorflow session
    tensorflow::SessionOptions options;
    tensorflow::ConfigProto & config = options.config;
    config.set_allow_soft_placement(true);
    auto session = NewSession(options);
    if (session == nullptr) {
        throw runtime_error("Could not create Tensorflow session.");
    }

    Status status;
    // Read in the protobuf graph we exported
    MetaGraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
    if (!status.ok()) {
        throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
    }

    // Add the graph to the session
    status = session->Create(graph_def.graph_def());
    if (!status.ok()) {
        throw runtime_error("Error creating graph: " + status.ToString());
    }

    // Read weights from the saved checkpoint
    Tensor checkpointPathTensor(DT_STRING, TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
    status = session->Run(
            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);
    if (!status.ok()) {
        throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
    }

    //Repeat (for each episode):
    //This is going to be interrupted by the ALE code since I set max_num_frames beforehand
    for (int episode = episodePassed + 1; totalNumberFrames < totalNumberOfFramesToLearn; episode++) {
        //firstState = ale.cloneState();
        //random no-op
        unsigned int noOpNum = 0;
        if (randomNoOp) {
            noOpNum = (*agentRand)() % (noOpMax) + 1;
            for (int i = 0; i < noOpNum; ++i) {
                ale.act(actions[0]);
            }
        }

        //We have to clean the traces every episode:
        for (unsigned int a = 0; a < nonZeroElig.size(); a++) {
            for (unsigned long long i = 0; i < nonZeroElig[a].size(); i++) {
                long long idx = nonZeroElig[a][i];
                e[a][idx] = 0.0;
            }
            nonZeroElig[a].clear();
        }


        F.clear();
        features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
        trueFeatureSize = F.size();
        groupFeatures(F);
        updateQValues(F, Q);

        currentAction = epsilonGreedy(Q, episode);
        gettimeofday(&tvBegin, NULL);
        int lives = ale.lives();
        //Repeat(for each step of episode) until game is over:
        //This also stops when the maximum number of steps per episode is reached
        while (!ale.game_over()) {
            reward.clear();
            reward.push_back(0.0);
            reward.push_back(0.0);
            updateQValues(F, Q);
            updateReplTrace(currentAction, F);

            sanityCheck();
            //Take action, observe reward and next state:
            act(ale, currentAction, reward);
            stepCount += 1;
            cumReward += reward[1];
            if (!ale.game_over()) {
                //Obtain active features in the new state:
                Fnext.clear();
                features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fnext);
                trueFnextSize = Fnext.size();
                groupFeatures(Fnext);
                updateQValues(Fnext, Qnext);     //Update Q-values for the new active features
                nextAction = epsilonGreedy(Qnext, episode);
                if (bufferIndex > actualBufferSize && planBufferSize < 0) {
                   if ((float) rand()/RAND_MAX < (float) actualBufferSize / bufferIndex) {
                       int randIndex = rand() % actualBufferSize;
                       stateBuffer[randIndex] = ale.cloneState();
                       screenBuffer[randIndex] = ale.cloneScreen();
                       rewardBuffer[randIndex] = reward[1];
                       prevBlobsBuffer[randIndex] = features->getPrevBlobs();
                       prevBlobsActiveColorsBuffer[randIndex] = features->getPrevBlobActiveColors();
                   }
                } else {
                    stateBuffer[bufferIndex % actualBufferSize] = ale.cloneState();
                    screenBuffer[bufferIndex % actualBufferSize] = ale.cloneScreen();
                    rewardBuffer[bufferIndex % actualBufferSize] = reward[1];
                    prevBlobsBuffer[bufferIndex % actualBufferSize] = features->getPrevBlobs();
                    prevBlobsActiveColorsBuffer[bufferIndex % actualBufferSize] = features->getPrevBlobActiveColors();
                }
                bufferIndex += 1;
            } else {
                nextAction = 0;
                for (unsigned int i = 0; i < Qnext.size(); i++) {
                    Qnext[i] = 0;
                }
            }
            //To ensure the learning rate will never increase along
            //the time, Marc used such approach in his JAIR paper
            if (trueFeatureSize > maxFeatVectorNorm) {
                maxFeatVectorNorm = trueFeatureSize;
                learningRate = alpha / maxFeatVectorNorm;
            }
            delta = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];

            //Update weights vector:
            for (unsigned int a = 0; a < nonZeroElig.size(); a++) {
                for (unsigned int i = 0; i < nonZeroElig[a].size(); i++) {
                    long long idx = nonZeroElig[a][i];
                    w[a][idx] = w[a][idx] + learningRate * delta * e[a][idx];
                }
            }
            // Planning step
            if (bufferIndex > maxBufferSize && totalNumberFrames < totalNumberOfFramesToLearn) {
                ale.saveState();
                prevBlobsSave = features->getPrevBlobs();
                prevBlobsActiveColorsSave = features->getPrevBlobActiveColors();
                for (int n = 0; n < planningIterations; n++) {
//                    oldWeights.clear();
//                    oldWeights = w;
                    int idx = rand() % (actualBufferSize - 4);
                    ALEState state = stateBuffer[idx];
                    ALEScreen screen = screenBuffer[idx];
                    ale.restoreState(state);
                    ale.restoreScreen(screen);
                    features->setPrevBlobs(prevBlobsBuffer[idx]);
                    features->setPrevBlobActiveColors(prevBlobsActiveColorsBuffer[idx]);

                    // Prime history
                    std::vector<std::vector<unsigned char>> rawFrameHistoryVec(4);
                    ale.getScreenRGB(rawFrameHistoryVec[0]);
                    std::vector<float> rewardHistoryVec(3);

                    for (int h = 1; h < 4; h++) {
                        Fplan.clear();
                        features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fplan);
                        groupPlanFeatures(Fplan);
                        state = stateBuffer[idx+h];
                        ale.restoreState(state);
                        screen = screenBuffer[idx+h];
                        ale.restoreScreen(screen);
                        ale.getScreenRGB(rawFrameHistoryVec[h]);
                        rewardHistoryVec[h-1] = rewardBuffer[idx+h];
                    }

                    int height = 210;
                    int width = 160;

                    Tensor frame_history(DT_FLOAT, TensorShape({1,210,160,12}));
                    for (int h = 0; h < 4; h++) {
                        for (int y = 0; y < height; y++) {
                            for (int x = 0; x < width; x++) {
                                frame_history.tensor<float, 4>()(0, y, x, h*3+0) =
                                        ((float) rawFrameHistoryVec[h][3 * width * y + 3 * x + 0] - backgroundChannelMeans[0]) / 255.0;
                                frame_history.tensor<float, 4>()(0, y, x, h*3+1) =
                                        ((float) rawFrameHistoryVec[h][3 * width * y + 3 * x + 1] - backgroundChannelMeans[1]) / 255.0;
                                frame_history.tensor<float, 4>()(0, y, x, h*3+2) =
                                        ((float) rawFrameHistoryVec[h][3 * width * y + 3 * x + 2] - backgroundChannelMeans[2])/ 255.0;
                            }
                        }
                    }


                    Tensor reward_history(DT_FLOAT, TensorShape({1, 3}));
                    for (int h = 0; h < 3; h++) {
                        reward_history.tensor<float, 2>()(0, h) = rewardHistoryVec[h];
                    }

                    Tensor actions(DT_FLOAT, TensorShape({1, 1, ale.getMinimalActionSet().size()}));
                    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
                            { "frame_history", frame_history },
                            { "reward_history", reward_history },
                            { "actions", actions },
                    };

                    // Clean plan traces
                    for (unsigned int a = 0; a < planNonZeroElig.size(); a++) {
                        for (unsigned long long i = 0; i < planNonZeroElig[a].size(); i++) {
                            long long idx = planNonZeroElig[a][i];
                            ePlan[a][idx] = 0.0;
                        }
                        planNonZeroElig[a].clear();
                    }

                    //cout << "Planning step " << n;
                    int k = 0;
                    int rewardSum = 0;

                    Fplan.clear();
                    features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fplan);
                    trueFeatureSize = Fplan.size();
                    groupPlanFeatures(Fplan);
                    updateQValues(Fplan, Q);

                    currentPlanAction = epsilonGreedy(Q, episode);
                    
                    actionFile<<totalNumberFrames<<std::endl;
                    while (!ale.game_over() && k < planningSteps) {
                        k += 1;

                        reward.clear();
                        reward.push_back(0.0);
                        reward.push_back(0.0);
                        updateQValues(Fplan, Q);
//                        updateQValuesWithWeights(Fplan, oldWeightsQ, oldWeights);
//                        actionFile<<Mathematics::argmax(Q)<<" "<<Mathematics::argmax(oldWeightsQ)<<std::endl;
                        updatePlanReplTrace(currentPlanAction, Fplan);

                        //sanityCheck();
                        //Take action, observe reward and next state:
//                        act(ale, currentPlanAction, reward);
                        oneHot(actions, ale.getMinimalActionSet().size(), currentPlanAction);
//                        std::cout<<currentPlanAction<<std::endl;

                        // The session will initialize the outputs
                        std::vector<tensorflow::Tensor> outputs;

                        // Run the session
                        status = session->Run({inputs},
                                              {"prediction_model/transform/transform/conv10/prediction_model/transform/conv10:0",
                                               "prediction_model/transform/transform/fc_reward_dec/prediction_model/transform/fc_reward_dec:0"},
                                              {}, &outputs);
                        if (!status.ok()) {
                            std::cout << status.ToString() << "\n";
                            throw runtime_error(status.ToString());
                        }

                        auto image_float = outputs[0].tensor<float,4>();
                        auto plan_reward = outputs[1].scalar<float>();

                        ALEScreen predicted_screen(210,160);
                        std::vector<unsigned char> rgbScreen(210*160*3);

                        for (int y = 0; y < height; y++) {
                            for (int x = 0; x < width; x++) {
                                rgbScreen[3 * width * y + 3 * x + 0] = (unsigned char) std::max(std::min((image_float(0, y, x, 0) * 255.0 + backgroundChannelMeans[0]), 255.0), 0.0);
                                rgbScreen[3 * width * y + 3 * x + 1] = (unsigned char) std::max(std::min((image_float(0, y, x, 1) * 255.0 + backgroundChannelMeans[1]), 255.0), 0.0);
                                rgbScreen[3 * width * y + 3 * x + 2] = (unsigned char) std::max(std::min((image_float(0, y, x, 2) * 255.0 + backgroundChannelMeans[2]), 255.0), 0.0);
                            }
                        }

                        ale.getALEScreenFromRGB(rgbScreen, predicted_screen, palette);

                        stepCount += 1;
                        if (!ale.game_over()) {
                            //Obtain active features in the new state:
                            FnextPlan.clear();
                            features->getActiveFeaturesIndices(predicted_screen, ale.getRAM(), FnextPlan);
                            trueFnextSize = FnextPlan.size();
                            groupPlanFeatures(FnextPlan);
                            updateQValues(FnextPlan, Qnext);     //Update Q-values for the new active features
                            nextPlanAction = epsilonGreedy(Qnext, episode);
                        } else {
                            nextPlanAction = 0;
                            for (unsigned int i = 0; i < Qnext.size(); i++) {
                                Qnext[i] = 0;
                            }
                        }
                        //To ensure the learning rate will never increase along
                        //the time, Marc used such approach in his JAIR paper
                        if (trueFeatureSize > maxFeatVectorNorm) {
                            maxFeatVectorNorm = trueFeatureSize;
                            learningRate = alpha / maxFeatVectorNorm;
                        }
                        delta = (float)plan_reward(0) + gamma * Qnext[nextPlanAction] - Q[currentPlanAction];

                        //Update weights vector:
                        for (unsigned int a = 0; a < planNonZeroElig.size(); a++) {
                            for (unsigned int i = 0; i < planNonZeroElig[a].size(); i++) {
                                long long idx = planNonZeroElig[a][i];
                                w[a][idx] = w[a][idx] + learningRate * delta * ePlan[a][idx];
                            }
                        }

                        Fplan = FnextPlan;
                        trueFeatureSize = trueFnextSize;
                        currentPlanAction = nextPlanAction;
                    }
                }
                ale.loadState();
                features->setPrevBlobs(prevBlobsSave);
                features->setPrevBlobActiveColors(prevBlobsActiveColorsSave);

            } // End planning
            F = Fnext;
            trueFeatureSize = trueFnextSize;
            currentAction = nextAction;
        }
        gettimeofday(&tvEnd, NULL);
        timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
        elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec) / 1000000.0;

        double fps = double(ale.getEpisodeFrameNumber()) / elapsedTime;
        printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps,\tlearning rate: %f\n",
               episode, cumReward - prevCumReward, (double) cumReward / (episode),
               ale.getEpisodeFrameNumber(), fps, learningRate);
        episodeResults.push_back(cumReward - prevCumReward);
        episodeFrames.push_back(ale.getEpisodeFrameNumber());
        episodeFps.push_back(fps);
        totalNumberFrames += stepCount * numStepsPerAction;
        stepCount = 0;
        prevCumReward = cumReward;
        features->clearCash();
        ale.reset_game();
        if (toSaveCheckPoint && totalNumberFrames > saveThreshold) {
            saveCheckPoint(episode, totalNumberFrames, episodeResults, saveWeightsEveryXFrames, episodeFrames,
                           episodeFps);
            saveThreshold += saveWeightsEveryXFrames;
        }
    }
    actionFile.close();
}

void SarsaLearner::evaluatePolicy(ALEInterface &ale, BlobTimeFeatures *features) {
    double reward = 0;
    double cumReward = 0;
    double prevCumReward = 0;
    struct timeval tvBegin, tvEnd, tvDiff;
    double elapsedTime;

    std::string oldName = checkPointName + "-Result-writing.txt";
    std::string newName = checkPointName + "-Result-finished.txt";
    std::ofstream resultFile;
    resultFile.open(oldName.c_str());

    std::string recordPath = "/local/ssd/gholland/" + checkPointName + "_record/";

    std::string actionRewardName = recordPath + "action-reward.txt";
    std::ofstream actionRewardFile;
    actionRewardFile.open(actionRewardName.c_str());

    int episode = 0;
    //Repeat (for each episode):
    for (int count = 0; episode < numEpisodesEval;) {
        episode++;
        //Repeat(for each step of episode) until game is over:
        gettimeofday(&tvBegin, NULL);
        //random no-op
        unsigned int noOpNum;
        if (randomNoOp) {
            noOpNum = (*agentRand)() % (noOpMax) + 1;
            for (int i = 0; i < noOpNum; ++i) {
                ale.act(actions[0]);
            }
        }
        for (int step = 0; !ale.game_over() && step < episodeLength; step++) {
            //Get state and features active on that state:
            F.clear();
            features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
            groupFeatures(F);
            updateQValues(F, Q);       //Update Q-values for each possible action
            currentAction = epsilonGreedy(Q);
            //Take action, observe reward and next state:
            //ale.saveScreenPNG(recordPath + to_string(static_cast<long long>(count)) + ".png");
            reward = ale.act(actions[currentAction]);
            //actionRewardFile<<actions[currentAction]<<" "<<reward<<" "<<ale.game_over()<<std::endl;
            count++;
            cumReward += reward;
        }
        //  ale.saveScreenPNG(recordPath + to_string(static_cast<long long>(count)) + ".png");
        gettimeofday(&tvEnd, NULL);
        timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
        elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec) / 1000000.0;
        double fps = double(ale.getEpisodeFrameNumber()) / elapsedTime;

        resultFile<<"Episode "<<episode<<": "<<cumReward-prevCumReward<<std::endl;
        printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n",
               episode, (cumReward - prevCumReward), (double) cumReward / (episode), ale.getEpisodeFrameNumber(), fps);
        features->clearCash();
        ale.reset_game();
        prevCumReward = cumReward;
    }
    resultFile << "Average: " << (double) cumReward / (episode) << std::endl;
    actionRewardFile.close();
    resultFile.close();
    rename(oldName.c_str(), newName.c_str());

}

void SarsaLearner::groupFeatures(vector<long long> &activeFeatures) {
    vector<long long> activeGroupIndices;

    int newGroup = 0;
    for (unsigned long long i = 0; i < activeFeatures.size(); ++i) {
        long long featureIndex = activeFeatures[i];
        if (featureTranslate[featureIndex] == 0) {
            if (newGroup) {
                featureTranslate[featureIndex] = numGroups;
                groups[numGroups - 1].numFeatures += 1;
            } else {
                newGroup = 1;
                Group agroup;
                agroup.numFeatures = 1;
                agroup.features.clear();
                groups.push_back(agroup);
                for (unsigned int action = 0; action < w.size(); ++action) {
                    w[action].push_back(0.0);
                    oldWeights[action].push_back(0.0);
                    e[action].push_back(0.0);
                    ePlan[action].push_back(0.0);
                }
                ++numGroups;
                featureTranslate[featureIndex] = numGroups;
            }
        } else {
            long long groupIndex = featureTranslate[featureIndex] - 1;
            auto it = &groups[groupIndex].features;
            if (it->size() == 0) {
                activeGroupIndices.push_back(groupIndex);
            }
            it->push_back(featureIndex);
        }
    }

    activeFeatures.clear();
    if (newGroup) {
        activeFeatures.push_back(groups.size() - 1);
    }

    for (unsigned long long index = 0; index < activeGroupIndices.size(); ++index) {
        long long groupIndex = activeGroupIndices[index];
        if (groups[groupIndex].features.size() != groups[groupIndex].numFeatures &&
            groups[groupIndex].features.size() != 0) {
            Group agroup;
            agroup.numFeatures = groups[groupIndex].features.size();
            agroup.features.clear();
            groups.push_back(agroup);
            ++numGroups;
            for (unsigned long long i = 0; i < groups[groupIndex].features.size(); ++i) {
                featureTranslate[groups[groupIndex].features[i]] = numGroups;
            }
            activeFeatures.push_back(numGroups - 1);
            for (unsigned a = 0; a < w.size(); ++a) {
                w[a].push_back(w[a][groupIndex]);
                oldWeights[a].push_back(oldWeights[a][groupIndex]);
                e[a].push_back(e[a][groupIndex]);
                ePlan[a].push_back(e[a][groupIndex]);
                if (e[a].back() >= traceThreshold) {
                    nonZeroElig[a].push_back(numGroups - 1);
                }
            }
            groups[groupIndex].numFeatures = groups[groupIndex].numFeatures - groups[groupIndex].features.size();
        } else if (groups[groupIndex].features.size() == groups[groupIndex].numFeatures) {
            activeFeatures.push_back(groupIndex);
        }
        groups[groupIndex].features.clear();
//        groups[groupIndex].features.shrink_to_fit();
    }
}

void SarsaLearner::groupPlanFeatures(vector<long long> &activeFeatures) {
    vector<long long> activeGroupIndices;

    int newGroup = 0;
    for (unsigned long long i = 0; i < activeFeatures.size(); ++i) {
        long long featureIndex = activeFeatures[i];
        if (featureTranslate[featureIndex] == 0) {
            if (newGroup) {
                featureTranslate[featureIndex] = numGroups;
                groups[numGroups - 1].numFeatures += 1;
            } else {
                newGroup = 1;
                Group agroup;
                agroup.numFeatures = 1;
                agroup.features.clear();
                groups.push_back(agroup);
                for (unsigned int action = 0; action < w.size(); ++action) {
                    w[action].push_back(0.0);
                    oldWeights[action].push_back(0.0);
                    e[action].push_back(0.0);
                    ePlan[action].push_back(0.0);
                }
                ++numGroups;
                featureTranslate[featureIndex] = numGroups;
            }
        } else {
            long long groupIndex = featureTranslate[featureIndex] - 1;
            auto it = &groups[groupIndex].features;
            if (it->size() == 0) {
                activeGroupIndices.push_back(groupIndex);
            }
            it->push_back(featureIndex);
        }
    }

    activeFeatures.clear();
    if (newGroup) {
        activeFeatures.push_back(groups.size() - 1);
    }

    for (unsigned long long index = 0; index < activeGroupIndices.size(); ++index) {
        long long groupIndex = activeGroupIndices[index];
        if (groups[groupIndex].features.size() != groups[groupIndex].numFeatures &&
            groups[groupIndex].features.size() != 0) {
            Group agroup;
            agroup.numFeatures = groups[groupIndex].features.size();
            agroup.features.clear();
            groups.push_back(agroup);
            ++numGroups;
            for (unsigned long long i = 0; i < groups[groupIndex].features.size(); ++i) {
                featureTranslate[groups[groupIndex].features[i]] = numGroups;
            }
            activeFeatures.push_back(numGroups - 1);
            for (unsigned a = 0; a < w.size(); ++a) {
                w[a].push_back(w[a][groupIndex]);
                oldWeights[a].push_back(oldWeights[a][groupIndex]);
                e[a].push_back(e[a][groupIndex]);
                ePlan[a].push_back(e[a][groupIndex]);
                if (ePlan[a].back() >= traceThreshold) {
                    planNonZeroElig[a].push_back(numGroups - 1);
                }
            }
            groups[groupIndex].numFeatures = groups[groupIndex].numFeatures - groups[groupIndex].features.size();
        } else if (groups[groupIndex].features.size() == groups[groupIndex].numFeatures) {
            activeFeatures.push_back(groupIndex);
        }
        groups[groupIndex].features.clear();
    }
}

void SarsaLearner::saveWeightsToFile(string suffix) {
    std::ofstream weightsFile((nameWeightsFile + suffix).c_str());
    if (weightsFile.is_open()) {
        weightsFile << w.size() << " " << w[0].size() << std::endl;
        for (unsigned int i = 0; i < w.size(); i++) {
            for (unsigned int j = 0; j < w[i].size(); j++) {
                if (w[i][j] != 0) {
                    weightsFile << i << " " << j << " " << w[i][j] << std::endl;
                }
            }
        }
        weightsFile.close();
    } else {
        printf("Unable to open file to write weights.\n");
    }
}

void SarsaLearner::loadWeights() {
    string line;
    int nActions, nFeatures;
    int i, j;
    double value;

    std::ifstream weightsFile(pathWeightsFileToLoad.c_str());

    weightsFile >> nActions >> nFeatures;
    assert(nActions == numActions);
    assert(nFeatures == numFeatures);

    while (weightsFile >> i >> j >> value) {
        w[i][j] = value;
    }
}
