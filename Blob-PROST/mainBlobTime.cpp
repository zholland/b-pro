/****************************************************************************************
** Starting point for running Sarsa algorithm. Here the parameters are set, the algorithm
** is started, as well as the features used. In fact, in order to create a new learning
** algorithm, once its class is implementend, the main file just need to instantiate
** Parameters, the Learner and the type of Features to be used. This file is a good 
** example of how to do it. A parameters file example can be seen in ../conf/sarsa.cfg.
** This is an example for other people to use: Sarsa with Basic Features.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef ALE_INTERFACE_H
#define ALE_INTERFACE_H
#include <ale_interface.hpp>
#endif
#ifndef PARAMETERS_H
#define PARAMETERS_H
#include "common/Parameters.hpp"
#endif
#ifndef SARSA_H
#define SARSA_H
#include "agents/rl/sarsa/SarsaLearner.hpp"
#endif
#ifndef BASIC_H
#define BASIC_H
#include "features/BlobTimeFeatures.hpp"
#endif

void printBasicInfo(Parameters param){
	printf("Seed: %d\n", param.getSeed());
	printf("\nCommand Line Arguments:\nPath to Config. File: %s\nPath to ROM File: %s\nPath to Backg. File: %s\n", 
		param.getConfigPath().c_str(), param.getRomPath().c_str(), param.getPathToBackground().c_str());
	if(param.getSubtractBackground()){
		printf("\nBackground will be subtracted...\n");
	}
	printf("\nParameters read from Configuration File:\n");
	printf("alpha:   %f\ngamma:   %f\nepsilon: %f\nlambda:  %f\nep. length: %d\n\n", 
		param.getAlpha(), param.getGamma(), param.getEpsilon(), param.getLambda(), 
		param.getEpisodeLength());
}


int main(int argc, char** argv){
    /*
// Initialize a tensorflow session
    tensorflow::SessionOptions options;
    tensorflow::ConfigProto & config = options.config;

    config.set_allow_soft_placement(true);

//    Session *session;
//    Status status = NewSession(options, &session);
//    if (!status.ok()) {
//        std::cout << status.ToString() << "\n";
//        return 1;
//    }

    const string pathToGraph = "/home/zach/data_and_checkpoints/reward_fixed_hist_128/checkpoints_seaquest_fulldecode/checkpoint_inference.meta";
    const string checkpointPath = "/home/zach/data_and_checkpoints/reward_fixed_hist_128/checkpoints_seaquest_fulldecode/checkpoint_inference";

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

    unsigned width, height;
    std::cout<<"Preparing inputs"<<std::endl;
    Tensor frame_history(DT_FLOAT, TensorShape({1,210,160,12}));
    std::vector<unsigned char> load_image; //the raw pixels
    unsigned decode_error = lodepng::decode(load_image, width, height, "/home/zach/b-pro/Blob-PROST/1_target_0.png");
    if(decode_error) std::cout << "decoder error " << decode_error << ": " << lodepng_error_text(decode_error) << std::endl;

//    for (int x,y = 0; i < load_image.size(); i+=4, j+=3) {
//        frame_history.tensor<float,4>()(0, 0, 0, 0) = image[i+0];
//    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int i = 0; i < 12; i++) {
//            frame_history.tensor<float,4>()(0, y, x, 0) = (float)load_image[4 * width * y + 4 * x + 0] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 1) = (float)load_image[4 * width * y + 4 * x + 1] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 2) = (float)load_image[4 * width * y + 4 * x + 2] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 3) = (float)load_image[4 * width * y + 4 * x + 0] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 4) = (float)load_image[4 * width * y + 4 * x + 1] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 5) = (float)load_image[4 * width * y + 4 * x + 2] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 6) = (float)load_image[4 * width * y + 4 * x + 0] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 7) = (float)load_image[4 * width * y + 4 * x + 1] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 8) = (float)load_image[4 * width * y + 4 * x + 2] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 9) = (float)load_image[4 * width * y + 4 * x + 0] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 10) = (float)load_image[4 * width * y + 4 * x + 1] / 255.0;
//            frame_history.tensor<float,4>()(0, y, x, 11) = (float)load_image[4 * width * y + 4 * x + 2] / 255.0;

                frame_history.tensor<float, 4>()(0, y, x, i) = 0.0;
            }
        }
    }

    Tensor reward_history(DT_FLOAT, TensorShape({1, 3}));
    reward_history.tensor<float, 2>()(0,0) = 0.0;
    reward_history.tensor<float, 2>()(0,1) = 0.0;
    reward_history.tensor<float, 2>()(0,2) = 0.0;

    Tensor actions(DT_FLOAT, TensorShape({1, 1, 18}));
    std::cout<<actions.NumElements()<<std::endl;
    actions.tensor<float, 3>()(0,0,0) = 0.0;
    actions.tensor<float, 3>()(0,0,1) = 0.0;
    actions.tensor<float, 3>()(0,0,2) = 0.0;
    actions.tensor<float, 3>()(0,0,3) = 0.0;
    actions.tensor<float, 3>()(0,0,4) = 0.0;
    actions.tensor<float, 3>()(0,0,5) = 0.0;
    actions.tensor<float, 3>()(0,0,6) = 0.0;
    actions.tensor<float, 3>()(0,0,7) = 0.0;
    actions.tensor<float, 3>()(0,0,8) = 0.0;
    actions.tensor<float, 3>()(0,0,9) = 0.0;
    actions.tensor<float, 3>()(0,0,10) = 0.0;
    actions.tensor<float, 3>()(0,0,11) = 0.0;
    actions.tensor<float, 3>()(0,0,12) = 0.0;
    actions.tensor<float, 3>()(0,0,13) = 0.0;
    actions.tensor<float, 3>()(0,0,14) = 0.0;
    actions.tensor<float, 3>()(0,0,15) = 0.0;
    actions.tensor<float, 3>()(0,0,16) = 0.0;
    actions.tensor<float, 3>()(0,0,17) = 0.0;

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            { "frame_history", frame_history },
            { "reward_history", reward_history },
            { "actions", actions },
    };

    // The session will initialize the outputs
    std::vector<tensorflow::Tensor> outputs;

    std::cout<<"Running ops"<<std::endl;
    // Run the session, evaluating our "c" operation from the graph
    status = session->Run({inputs},
                          {"prediction_model/transform/transform/conv10/prediction_model/transform/conv10:0",
                           "prediction_model/transform/transform/fc_reward_dec/prediction_model/transform/fc_reward_dec:0"},
                          {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }

    std::cout<<"Collecting output"<<std::endl;
    // Grab the first output (we only evaluated one graph node: "c")
    // and convert the node to a scalar representation.
    auto image_float = outputs[0].tensor<float,4>();
//    std::cout<<image<<std::endl;

    std::vector<unsigned char> image(210*160*4);

    height = 210;
    width = 160;

//    for (int y = 0; y < height; y++) {
//        for (int x = 0; x < width; x++) {
//            image[4 * width * y + 4 * x + 0] = (unsigned char) (image_float(0, y, x, 0) * 255.0);
//            image[4 * width * y + 4 * x + 1] = (unsigned char) (image_float(0, y, x, 1) * 255.0);
//            image[4 * width * y + 4 * x + 2] = (unsigned char) (image_float(0, y, x, 2) * 255.0);
//            image[4 * width * y + 4 * x + 3] = (unsigned char) 255;
//        }
//    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image[4 * width * y + 4 * x + 0] = (unsigned char) std::max(std::min((outputs[0].tensor<float,4>()(0, y, x, 0) * 255.0 + 32.0694), 255.0), 0.0);
            image[4 * width * y + 4 * x + 1] = (unsigned char) std::max(std::min((outputs[0].tensor<float,4>()(0, y, x, 1) * 255.0 + 44.1966), 255.0), 0.0);
            image[4 * width * y + 4 * x + 2] = (unsigned char) std::max(std::min((outputs[0].tensor<float,4>()(0, y, x, 2) * 255.0 + 111.029), 255.0), 0.0);
            image[4 * width * y + 4 * x + 3] = (unsigned char) 255;
        }
    }

    //Encode the image
    unsigned encode_error = lodepng::encode("inference.png", image, width, height);

    //if there's an error, display it
    if(encode_error) std::cout << "encoder error " << encode_error << ": "<< lodepng_error_text(encode_error) << std::endl;


    auto reward = outputs[1].scalar<float>();

    // (There are similar methods for vectors and matrices here:
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

    // Print the results
    std::cout << outputs[1].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
    std::cout << reward() << "\n"; // 30

    // Free any resources used by the session
    session->Close();
    */
////Reading parameters from file defined as input in the run command:
//    Parameters param(argc, argv);
//    srand(param.getSeed());
//
////Using Basic features:
//    BlobTimeFeatures features(&param);
////Reporting parameters read:
//    printBasicInfo(param);
//
//    ALEInterface ale(0);
//
//    ale.setFloat("repeat_action_probability", 0.25);
//    ale.setInt("random_seed", 2 * param.getSeed());
//    ale.setInt("frame_skip", param.getNumStepsPerAction());
//    ale.setInt("max_num_frames_per_episode", param.getEpisodeLength());
//    ale.setBool("color_averaging", true);
//
//    ale.loadROM(param.getRomPath().c_str());
//
//    //mt19937 agentRand(param.getSeed());
//    //Instantiating the learning algorithm:
//    //	SarsaLearner sarsaLearner(ale, &features, &param, 2*param.getSeed()-1);
//    //Learn a policy:
//    //    sarsaLearner.learnPolicy(ale, &features);
//    ActionVect actions = ale.getMinimalActionSet();
//    ale.reset_game();
////	ale.act(actions[10]);
////	ale.act(actions[11]);
////	ale.act(actions[12]);
////	ale.act(actions[13]);
//vector<long long> F;
//for (int k = 0; k < 25; k++) {
////        image.clear();
////        std::cout<<k<<std::endl;
//  ALEScreen screen(210, 160);
//  unsigned width, height;
//  std::vector<unsigned char> rgbScreen(210*160*3);
//  std::vector<unsigned char> image; //the raw pixels
//  unsigned error = lodepng::decode(image, width, height, "/home/zach/b-pro/Blob-PROST/1_target_" + std::to_string(k) + ".png");
//  if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
//  for (int i,j = 0; i < image.size(); i+=4, j+=3) {
//      rgbScreen[j+0] = image[i+0];
//      rgbScreen[j+1] = image[i+1];
//      rgbScreen[j+2] = image[i+2];
//  }
//  ale.getALEScreenFromRGB(rgbScreen, screen);
//  F.clear();
//  features.getActiveFeaturesIndices(screen, ale.getRAM(), F);
//  std::cout << F.size() << std::endl;
//}
//for (int k = 0; k < 25; k++) {
////        image.clear();
////        std::cout<<k<<std::endl;
//  ALEScreen screen(210, 160);
//  unsigned width, height;
//  std::vector<unsigned char> rgbScreen(210*160*3);
//  std::vector<unsigned char> image; //the raw pixels
//  unsigned error = lodepng::decode(image, width, height, "/home/zach/b-pro/Blob-PROST/1_prediction_" + std::to_string(k) + ".png");
//  if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
//  for (int i,j = 0; i < image.size(); i+=4, j+=3) {
//      rgbScreen[j+0] = image[i+0];
//      rgbScreen[j+1] = image[i+1];
//      rgbScreen[j+2] = image[i+2];
//  }
//  ale.getALEScreenFromRGB(rgbScreen, screen);
//  F.clear();
//  features.getActiveFeaturesIndices(screen, ale.getRAM(), F);
//  std::cout << F.size() << std::endl;
//}
//    for (int k = 10; k < 25; k++) {
//        image.clear();
//        unsigned error = lodepng::decode(image, width, height, "/home/zach/b-pro/Blob-PROST/1_prediction_" + std::to_string(k) + ".png");
//        for (int i,j = 0; i < image.size(); i+=4, j+=3) {
//            rgbScreen[j+0] = image[i+0];
//            rgbScreen[j+1] = image[i+1];
//            rgbScreen[j+2] = image[i+2];
//        }
//        ale.getALEScreenFromRGB(rgbScreen, screen);
//        F.clear();
//        features.getActiveFeaturesIndices(screen, ale.getRAM(), F);
//        std::cout << F.size() << std::endl;
//    }
//	for (std::vector<long long>::const_iterator i = F.begin(); i != F.end(); ++i)
//		std::cout << *i << ' ';
//	std::cout<<std::endl;
//	std::cout<<(int)ale.getScreen().arraySize()<<std::endl;
//	std::cout<<(unsigned int)ale.getScreen().getArray()[20*160+21]<<std::endl;

//	ale.saveScreenPNG("test.png");
//	std::vector<unsigned char> image; //the raw pixels
//	unsigned width, height;
//
//	//decode
////	unsigned error = lodepng::decode(image, width, height, "/home/zach/b-pro/Blob-PROST/test.png");
//	unsigned error = lodepng::decode(image, width, height, "/home/zach/b-pro/Blob-PROST/1_prediction_10.png");
////	unsigned error = lodepng::decode(image, width, height, "/home/zach/b-pro/Blob-PROST/1_target_10.png");
//    std::cout<<"Height: "<<height<<", Width: "<<width<<std::endl;
//
////	//if there's an error, display it
////	if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
////	for (int i = 3; i < image.size(); i+=4) {
////		if ((int)image[i] == 255)
////			std::cout << (int) image[i] << std::endl;
////	}
////	//the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
//	ALEScreen screen(210, 160);
////	ALEScreen screen = ale.getScreen();
//	std::vector<unsigned char> rgbScreen(210*160*3);
//
//	for (int i,j = 0; i < image.size(); i+=4, j+=3) {
//		rgbScreen[j+0] = image[i+0];
//		rgbScreen[j+1] = image[i+1];
//		rgbScreen[j+2] = image[i+2];
//	}
//
////	ale.getScreenRGB(rgbScreen);
////	std::cout<<(int)ale.getScreen().getArray()[0]<<" "<<(int)screen.getArray()[2]<<std::endl;
////	std::cout<<screen.arraySize()<<std::endl;
//	ale.getALEScreenFromRGB(rgbScreen, screen);
////	ale.getALEScreenFromRGB(image, screen);
//	std::cout<<std::endl;
//	std::cout<<"Screen equals copied screen: "<<ale.getScreen().equals(screen)<<std::endl;
//    pixel_t* mainScreen = ale.getScreen().getArray();
//    pixel_t* copyScreen = screen.getArray();
//    int count = 0;
//    for (int i = 0; i < (210*160); i++) {
////        if (mainScreen[i] != copyScreen[i]) {
////            count++;
////            std::cout<<"Main: "<<(int)mainScreen[i]<<std::endl;
////            std::cout<<"Copy: "<<(int)copyScreen[i]<<std::endl;
//            mainScreen[i] = copyScreen[i];
////            mainScreen[i] = 100;
////        }
//    }
//    std::cout<<"Differences: "<<count<<std::endl;
////    ale.saveScreenPNG("test2.png");
//    ale.saveScreenPNG("ntsc_prediction.png");
////	std::memcpy(screen.getArray(), &image, screen.arraySize());
    //Reading parameters from file defined as input in the run command:
    Parameters param(argc, argv);
    srand(param.getSeed());

    //Using Basic features:
    BlobTimeFeatures features(&param);
    //Reporting parameters read:
    printBasicInfo(param);

//    ALEInterface ale(param.getDisplay());
    ALEInterface ale(0);

    ale.setFloat("repeat_action_probability", 0.25);
    ale.setInt("random_seed", 2*param.getSeed());
    ale.setInt("frame_skip", param.getNumStepsPerAction());
    ale.setInt("max_num_frames_per_episode", param.getEpisodeLength());
    ale.setBool("color_averaging", true);

    ale.loadROM(param.getRomPath().c_str());

    //mt19937 agentRand(param.getSeed());
    //Instantiating the learning algorithm:
    SarsaLearner sarsaLearner(ale, &features, &param, 2*param.getSeed()-1);
    //Learn a policy:
    sarsaLearner.learnPolicy(ale, &features);


    printf("\n\n== Evaluation without Learning == \n\n");
    sarsaLearner.evaluatePolicy(ale, &features);
    return 0;
}
