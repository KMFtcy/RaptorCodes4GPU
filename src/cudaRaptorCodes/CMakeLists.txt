#Compile all files in this directory into static libraries
aux_source_directory(. CUDA_R10)

add_library(cudaR10 ${CUDA_R10})
