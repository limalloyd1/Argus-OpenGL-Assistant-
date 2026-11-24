# Argus-OpenGL-Assistant-
An LLM trained on OpenGL documentation, specialized in answering questions related to optimizing and maintaining C++ OpenGL code 

Codebase is used to train model inferencing and set up trainset for Qwen 1.7B model. Unsloth workflow is recommended for outputting GGUF file for local interfacing  

Argus leverages qwen2.5:3b as a 'teacher' model which sets up the q&a data pipeline for the 'student' model 

Steps for training:
1. Generate QA -> sets up Question and Answer trainset
2. preprocess -> chunks dataset for more optimized learning
3. split_dataset -> organizes data set into columns for UnSloth training
