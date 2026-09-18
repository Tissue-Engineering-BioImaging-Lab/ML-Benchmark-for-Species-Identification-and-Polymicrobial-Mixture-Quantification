To run the Transformer notebooks on GPU (uses the shared raman-cnn-gpu image):

1:
.\run_gpu_jupyter.ps1
then http://localhost:8888/lab


2: .\train_gpu_headless.ps1 <file name>
   e.g. .\train_gpu_headless.ps1 transformer_single_noisy_separate.ipynb

If the image is missing, build it once from the CNN folder (where the Dockerfile lives):
   cd ..\CNN; docker build -f Dockerfile.gpu -t raman-cnn-gpu .
