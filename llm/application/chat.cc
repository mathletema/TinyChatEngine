#include <iostream>
#include <map>
#include <string>
#include <cstring>

#include "Generate.h"

std::map<std::string, int> model_config = {
    {"OPT_125m", OPT_125M},       {"OPT_1.3B", OPT_1_3B}, {"OPT_6.7B", OPT_6_7B},         {"LLaMA_7B", LLaMA_7B},
    {"LLaMA2_7B_chat", LLaMA_7B}, {"7b", LLaMA_7B},       {"LLaMA2_13B_chat", LLaMA_13B}, {"13b", LLaMA_13B},
    {"CodeLLaMA_7B_Instruct", CodeLLaMA_7B},   {"CodeLLaMA_13B_Instruct", CodeLLaMA_13B}, 
    {"StarCoder", StarCoder_15_5B}, {"StarCoder_15.5B", StarCoder_15_5B}
    };

std::map<std::string, std::string> model_path = {{"OPT_125m", "models/OPT_125m"},
                                                 {"OPT_1.3B", "models/OPT_1.3B"},
                                                 {"OPT_6.7B", "models/OPT_6.7B"},
                                                 {"LLaMA_7B", "models/LLaMA_7B"},
                                                 {"LLaMA2_7B_chat", "models/LLaMA_7B_2_chat"},
                                                 {"LLaMA2_13B_chat", "models/LLaMA_13B_2_chat"},
                                                 {"7b", "models/LLaMA_7B_2_chat"},
                                                 {"13b", "models/LLaMA_13B_2_chat"},
                                                 {"CodeLLaMA_7B_Instruct", "models/CodeLLaMA_7B_Instruct"},
                                                 {"CodeLLaMA_13B_Instruct", "models/CodeLLaMA_13B_Instruct"},
                                                 {"StarCoder", "models/StarCoder"},
                                                 {"StarCoder_15.5B", "models/StarCoder"}
                                                 };

std::map<std::string, int> data_format_list = {
    {"FP32", FP32}, {"INT8", QINT8}, {"INT4", INT4}, {"int4", INT4}, {"fp32", FP32},
};

bool isLLaMA(std::string s) {
    std::string LLaMA_prefix = "LLaMA";
    std::string CodeLLaMA_prefix = "CodeLLaMA";

    if (s.substr(0, LLaMA_prefix.size()) == LLaMA_prefix || s.substr(0, CodeLLaMA_prefix.size()) == CodeLLaMA_prefix || s == "7b" || s == "13b")
        return true;
    else
        return false;
}

bool isCodeLLaMA(std::string s) {
    std::string CodeLLaMA_prefix = "CodeLLaMA";

    if (s.substr(0, CodeLLaMA_prefix.size()) == CodeLLaMA_prefix)
        return true;
    else
        return false;
}

bool isStarCoder(std::string s) {
    std::string StarCoder_prefix = "StarCoder";

    if (s.substr(0, StarCoder_prefix.size()) == StarCoder_prefix)
        return true;
    else
        return false;
}

bool convertToBool(const char* str) {
    if (strcmp(str, "true") == 0 || strcmp(str, "1") == 0) {
        return true;
    }
    else if (strcmp(str, "false") == 0 || strcmp(str, "0") == 0) {
        return false;
    }
    else {
        std::cerr << "Error: Invalid boolean value: " << str << std::endl;
        exit(EXIT_FAILURE);
    }
}

int NUM_THREAD = 8;

int main(int argc, char* argv[]) {
    bool use_voicechat = false;

    // Check for optional arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-v") == 0) {
            use_voicechat = true;
            // Remove the flag from argc and argv
            for (int j = i; j < argc - 1; ++j) {
                argv[j] = argv[j + 1];
            }
            --argc;
            break;
        }
    }

    std::string target_model = "LLaMA2_7B_chat";
    std::string target_data_format = "INT4";
    bool instruct = true;
    Profiler::getInstance().for_demo = true;

    std::cout << "TinyChatEngine by MIT HAN Lab: https://github.com/mit-han-lab/TinyChatEngine" << std::endl;
    if (argc >= 3 && argc <= 5) {
        if (argc >= 4) {
            NUM_THREAD = atoi(argv[3]);
        }
        if (argc == 5) {
            instruct = convertToBool(argv[4]);
        }

        auto target_str = argv[1];
        target_model = argv[1];
        if (model_config.count(target_model) == 0) {
            std::cerr << "Model config:" << target_str << " unsupported" << std::endl;
            std::cerr << "Please select one of the following:";
            for (const auto& k : model_config) {
                std::cerr << k.first << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported model\n");
        }
        std::cout << "Using model: " << argv[1] << std::endl;

        auto data_format_input = argv[2];
        if (data_format_list.count(data_format_input) == 0) {
            std::cerr << "Data format:" << data_format_input << " unsupported" << std::endl;
            std::cerr << "Please select one of the following: ";
            for (const auto& k : data_format_list) {
                std::cerr << k.first << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported data format\n");
        }
        target_data_format = argv[2];
        if (target_data_format == "INT4" || target_data_format == "int4")
            std::cout << "Using AWQ for 4bit quantization: https://github.com/mit-han-lab/llm-awq" << std::endl;
        else
            std::cout << "Using data format: " << argv[2] << std::endl;
    } else if (argc == 2) {
        auto target_str = argv[1];
        target_model = argv[1];
        if (model_config.count(target_model) == 0) {
            std::cerr << "Model config:" << target_str << " unsupported" << std::endl;
            std::cerr << "Please select one of the following: ";
            for (const auto& k : model_config) {
                std::cerr << k.first << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported model\n");
        }
        std::cout << "Using model: " << argv[1] << std::endl;

        auto data_format_input = "INT4";
    } else {
        if (isLLaMA(target_model)) {
            std::cout << "Using model: " + target_model << std::endl;
            if (target_data_format == "INT4" || target_data_format == "int4")
                std::cout << "Using AWQ for 4bit quantization: https://github.com/mit-han-lab/llm-awq" << std::endl;
            else
                std::cout << "Using data format: " << target_data_format << std::endl;
        } 
        else if (isStarCoder(target_model)) {
            std::cout << "Using model: " + target_model << std::endl;
            if (target_data_format == "INT4" || target_data_format == "int4")
                std::cout << "Using AWQ for 4bit quantization: https://github.com/mit-han-lab/llm-awq" << std::endl;
            else
                std::cout << "Using data format: " << target_data_format << std::endl;
        }
        else {  // OPT
            target_model = "OPT6.7B";
            target_data_format = "INT8";
            std::cout << "Using model: " + target_model << std::endl;
            std::cout << "Using data format: " + target_data_format << std::endl;
        }
    }

    if (isLLaMA(target_model)) {
        int format_id = data_format_list[target_data_format];

        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = model_config[target_model];
        std::string m_path = model_path[target_model];

        #ifdef MODEL_PREFIX
        m_path = MODEL_PREFIX + m_path;
        #endif

        struct opt_params generation_config;
        generation_config.n_predict = 512;
        generation_config.repeat_penalty = 1.1f;
        generation_config.temp = 0.2f;
        if(isCodeLLaMA(target_model)) {
            generation_config.n_vocab = 32016;
        }
        else {
            generation_config.n_vocab = 32000;
        }

        bool first_prompt = true;

        if (format_id == FP32) {
            Fp32LlamaForCausalLM model = Fp32LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;

            // Get input from the user
            while (true) {
                std::string input;
                if (use_voicechat){
                    int result = std::system("./application/sts_utils/listen");
                    std::ifstream in("tmpfile");
                    std::getline(in, input);
                    result = std::system("rm tmpfile");
                    (void)result;
                    std::cout << input << std::endl;
                } else {
                    std::cout << "USER: ";
                    std::getline(std::cin, input);
                }
                if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                    break;
                if (instruct) {
                    std::cout << "ASSISTANT: " << std::endl;
                    if (isCodeLLaMA(target_model)) {
                        if (first_prompt) {
                            input = "<s>[INST] " + input + " [/INST] ";
                            first_prompt = false;
                        }
                        else {
                            input = " </s> <s>[INST] " + input + " [/INST] ";
                        }
                    }

                }
                else {
                    if (isCodeLLaMA(target_model)) {
                        std::cout << input;
                    }
                }

                if (!isCodeLLaMA(target_model)) {
                    if (first_prompt) {
                        input = "A chat between a human and an assistant.\n\n### Human: " + input + "\n### Assistant: \n";
                        first_prompt = false;
                    }
                    else {
                        input = "### Human: " + input + "\n### Assistant: \n";
                    }
                }

                LLaMAGenerate(m_path, &model, LLaMA_FP32, input, generation_config, "models/llama_vocab.bin", true, false);
            }
        } else if (format_id == INT4) {
            m_path = "INT4/" + m_path;
            Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;
            
            // Get input from the user
            while (true) {
                std::string input;
                if (use_voicechat){
                    int result = std::system("./application/sts_utils/listen");
                    std::ifstream in("tmpfile");
                    std::getline(in, input);
                    result = std::system("rm tmpfile");
                    (void)result;
                    std::cout << input << std::endl;
                } else {
                    std::cout << "USER: ";
                    std::getline(std::cin, input);
                }
                if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                    break;
                if (instruct) {
                    std::cout << "ASSISTANT: " << std::endl;
                    if (isCodeLLaMA(target_model)) {
                        if (first_prompt) {
                            input = "<s>[INST] " + input + " [/INST] ";
                            first_prompt = false;
                        }
                        else {
                            input = " </s> <s>[INST] " + input + " [/INST] ";
                        }
                    }
                }
                else {
                    if (isCodeLLaMA(target_model)) {
                        std::cout << input;
                    }
                }

                if (!isCodeLLaMA(target_model)) {
                    if (first_prompt) {
                        input = "A chat between a human and an assistant.\n\n### Human: " + input + "\n### Assistant: \n";
                        first_prompt = false;
                    }
                    else {
                        input = "### Human: " + input + "\n### Assistant: \n";
                    }
                }
                LLaMAGenerate(m_path, &model, LLaMA_INT4, input, generation_config, "models/llama_vocab.bin", true, use_voicechat);
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for LLaMA7B." << std::endl;
        }
    } else if (isStarCoder(target_model)) {
        int format_id = data_format_list[target_data_format];

        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = model_config[target_model];
        std::string m_path = model_path[target_model];

        #ifdef MODEL_PREFIX
        m_path = MODEL_PREFIX + m_path;
        #endif

        struct opt_params generation_config;
        generation_config.n_predict = 128;
        // generation_config.repeat_penalty = 1.1f;
        generation_config.top_k = 0;
        generation_config.temp = 0.2f;
        generation_config.n_vocab = 49152;

        if (format_id == FP32) {
            Fp32GPTBigCodeForCausalLM model = Fp32GPTBigCodeForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;

            // Get input from the user
            while (true) {
                std::cout << "USER: ";
                std::string input;
                std::getline(std::cin, input);
                std::cout << input;

                GPTBigCodeGenerate(m_path, &model, StarCoder_FP32, input, generation_config, "models/starcoder_vocab.bin", true);
            }
        } else if (format_id == INT4) {
            m_path = "INT4/" + m_path;
            Int4GPTBigCodeForCausalLM model = Int4GPTBigCodeForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;

            // Get input from the user
            while (true) {
                std::cout << "USER: ";
                std::string input;
                std::getline(std::cin, input);
                std::cout << input;

                GPTBigCodeGenerate(m_path, &model, StarCoder_INT4, input, generation_config, "models/starcoder_vocab.bin", true);    
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for StarCoder." << std::endl;
        }
    } else {  // OPT
#ifdef QM_CUDA
        printf("OPT is not supported with CUDA backend yet.");
        exit(-1);
#else
        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = model_config[target_model];
        std::string m_path = model_path[target_model];
        int format_id = data_format_list[target_data_format];

        // Load encoder
        std::string bpe_file = "models/opt_merges.txt";
        std::string vocab_file = "models/opt_vocab.json";
        Encoder encoder = get_encoder(vocab_file, bpe_file);
        std::string decode;

        struct opt_params generation_config;
        generation_config.n_predict = 512;
        if (format_id == QINT8) {
            OPTForCausalLM model = OPTForCausalLM("INT8/" + m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;
            
            // Get input from the user
            std::string input;
            if (use_voicechat){
                int result = std::system("./application/sts_utils/listen");
                std::ifstream in("tmpfile");
                std::getline(in, input);
                result = std::system("rm tmpfile");
                (void)result;
                std::cout << input << std::endl;
            } else {
                std::cout << "USER: ";
                std::getline(std::cin, input);
            }
            std::vector<int> input_ids = encoder.encode(input);
            std::string decoded = encoder.decode(input_ids);

            // Generate
            std::vector<int> generated_ids =
                OPTGenerate(&model, OPT_INT8, input_ids, generation_config, &encoder, true, use_voicechat);
        } else if (format_id == FP32) {
            Fp32OPTForCausalLM model = Fp32OPTForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;

            // Get input from the user
            std::string input;
            if (use_voicechat){
                int result = std::system("./application/sts_utils/listen");
                std::ifstream in("tmpfile");
                std::getline(in, input);
                result = std::system("rm tmpfile");
                (void)result;
                std::cout << input << std::endl;
            } else {
                std::cout << "USER: ";
                std::getline(std::cin, input);
            }
            std::vector<int> input_ids = encoder.encode(input);
            std::string decoded = encoder.decode(input_ids);

            // Generate
            std::vector<int> generated_ids =
                OPTGenerate(&model, OPT_FP32, input_ids, generation_config, &encoder, true, use_voicechat);
        } else if (format_id == INT4) {
            Int4OPTForCausalLM model = Int4OPTForCausalLM("INT4/" + m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;

            // Get input from the user
            std::string input;
            if (use_voicechat){
                int result = std::system("./application/sts_utils/listen");
                std::ifstream in("tmpfile");
                std::getline(in, input);
                result = std::system("rm tmpfile");
                (void)result;
                std::cout << input << std::endl;
            } else {
                std::cout << "USER: ";
                std::getline(std::cin, input);
            }
            
            std::vector<int> input_ids = encoder.encode(input);
            std::string decoded = encoder.decode(input_ids);

            // Generate
            std::vector<int> generated_ids =
                OPTGenerate(&model, OPT_INT4, input_ids, generation_config, &encoder, true, use_voicechat);
        }
#endif  // QN_CUDA
    }
};
