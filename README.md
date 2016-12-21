# convolutional-autoencoder-for-raw-waveform-reconstruction
convolutional autoencoder for raw waveform reconstruction to replace the classic STFT, i called it as short-time AE transform (STAET)

For now, it can reconstruct the raw waveforms of audio.
The convolution + pooling then deconvolution + upsampling

Pixels as features work very well for image processing, why not raw waveform for audio or speech processing? I know google Tara has some work of using CLDNN (or convolutional LSTM) to model on raw waveform for speech recognition. But it still very difficult especially on small datasets. Because all of the FIR filters should be learned from data.

Contact:
yx0001@surrey.ac.uk
