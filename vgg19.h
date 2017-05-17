#ifndef VGG19_H_
#define VGG19_H_

#include <cstdio>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

#include "jzq.h"

struct VGG19
{
  std::vector<A3f> conv1_1;
  std::vector<A3f> conv1_2;

  std::vector<A3f> conv2_1;
  std::vector<A3f> conv2_2;

  std::vector<A3f> conv3_1;
  std::vector<A3f> conv3_2;
  std::vector<A3f> conv3_3;
  std::vector<A3f> conv3_4;

  std::vector<A3f> conv4_1;
  std::vector<A3f> conv4_2;
  std::vector<A3f> conv4_3;
  std::vector<A3f> conv4_4;

  std::vector<A3f> conv5_1;
  std::vector<A3f> conv5_2;
  std::vector<A3f> conv5_3;
  std::vector<A3f> conv5_4;

  A2f fc6;
  A2f fc7;
  A2f fc8;
};

inline VGG19 vgg19loadFromDir(const std::string& path);

inline A3f vgg19forwardPass(const A2V3f& image, // float rgb image, [0..1]
                            const VGG19& vgg19,
                            const int    upToLayer=8);


////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////


inline A3f convolve(const A3f& input,const std::vector<A3f>& kernels)
{
  const int numKernels = kernels.size();

  A3f output(input.width(),input.height(),numKernels);

  #pragma omp parallel for
  for(int i=0;i<numKernels;i++)
  {
    const A3f& kernel = kernels[i];

    assert(kernel.depth()==input.depth());

    for(int y=0;y<input.height();y++)
    for(int x=0;x<input.width();x++)
    {
      float sum = 0;

      for(int kz=0;kz<kernel.depth();kz++)
      for(int ky=0;ky<kernel.height();ky++)
      for(int kx=0;kx<kernel.width();kx++)
      {
        const int ix = x+kx-kernel.width()/2;
        const int iy = y+ky-kernel.height()/2;

        if (ix>=0 && ix<input.width() &&
            iy>=0 && iy<input.height())
        {
          sum += kernel(kx,ky,kz) * input(ix,iy,kz);
        }
      }

      output(x,y,i) = sum;
    }
  }

  return output;
}

inline A3f relu(const A3f& input)
{
  A3f output(size(input));

  #pragma omp parallel for
  for(int xyz=0;xyz<numel(output);xyz++)
  {
    output[xyz] = std::max(input[xyz],0.0f);
  }

  return output;
}

inline A3f maxpool(const A3f& input)
{
  A3f output(input.width()/2,input.height()/2,input.depth());

  #pragma omp parallel for
  for(int z=0;z<output.depth();z++)
  {
    for(int y=0;y<output.height();y++)
    for(int x=0;x<output.width();x++)
    {
      output(x,y,z) = std::max(std::max(input(x*2  ,y*2  ,z),
                                        input(x*2+1,y*2  ,z)),
                               std::max(input(x*2  ,y*2+1,z),
                                        input(x*2+1,y*2+1,z)));
    }
  }

  return output;
}

inline A3f flatten(const A3f& input)
{
  A3f output(1,1,numel(input));

  #pragma omp parallel for
  for(int i=0;i<numel(input);i++)
  {
    output(0,0,i) = input[i];
  }

  return output;
}

inline A3f product(const A3f& input,const A2f& weights)
{
  assert(input.depth()==weights.width());

  A3f output(1,1,weights.height());

  #pragma omp parallel for
  for(int j=0;j<output.depth();j++)
  {
    float sum = 0;
    for(int i=0;i<input.depth();i++) { sum += weights(i,j) * input(0,0,i); }
    output(0,0,j) = sum;
  }

  return output;
}

inline A3f vgg19forwardPass(const A2V3f& image,const VGG19& vgg19,const int upToLayer)
{
  A3f X(image.width(),image.height(),3);

  for(int y=0;y<image.height();y++)
  for(int x=0;x<image.width();x++)
  {
    /// RGB -> BGR, [0..1] -> [0..255], subtract mean
    X(x,y,0) = image(x,y)[2]*255.0f - 103.939f;
    X(x,y,1) = image(x,y)[1]*255.0f - 116.779f;
    X(x,y,2) = image(x,y)[0]*255.0f - 123.68f;
  }

  if (upToLayer==0) { return X; }

  X = relu(convolve(X,vgg19.conv1_1)); printf("conv1_1\n"); if (upToLayer==1) { return X; }
  X = relu(convolve(X,vgg19.conv1_2)); printf("conv1_2\n");
  X = maxpool(X);

  X = relu(convolve(X,vgg19.conv2_1)); printf("conv2_1\n"); if (upToLayer==2) { return X; }
  X = relu(convolve(X,vgg19.conv2_2)); printf("conv2_2\n");
  X = maxpool(X);

  X = relu(convolve(X,vgg19.conv3_1)); printf("conv3_1\n"); if (upToLayer==3) { return X; }
  X = relu(convolve(X,vgg19.conv3_2)); printf("conv3_2\n");
  X = relu(convolve(X,vgg19.conv3_3)); printf("conv3_3\n");
  X = relu(convolve(X,vgg19.conv3_4)); printf("conv3_4\n");
  X = maxpool(X);

  X = relu(convolve(X,vgg19.conv4_1)); printf("conv4_1\n"); if (upToLayer==4) { return X; }
  X = relu(convolve(X,vgg19.conv4_2)); printf("conv4_2\n");
  X = relu(convolve(X,vgg19.conv4_3)); printf("conv4_3\n");
  X = relu(convolve(X,vgg19.conv4_4)); printf("conv4_4\n");
  X = maxpool(X);

  X = relu(convolve(X,vgg19.conv5_1)); printf("conv5_1\n"); if (upToLayer==5) { return X; }
  X = relu(convolve(X,vgg19.conv5_2)); printf("conv5_2\n");
  X = relu(convolve(X,vgg19.conv5_3)); printf("conv5_3\n");
  X = relu(convolve(X,vgg19.conv5_4)); printf("conv5_4\n");
  X = maxpool(X);

  X = flatten(X);
  X = relu(product(X,vgg19.fc6));      printf("fc6\n");
  X = relu(product(X,vgg19.fc7));      printf("fc7\n");
  X = product(X,vgg19.fc8);            printf("fc8\n");

  return X;
}

inline std::vector<A3f> loadKernels(const std::string& fileNameFormat,const int numKernels)
{
  std::vector<A3f> kernels(numKernels);
  for(int i=0;i<numKernels;i++)
  {
    const std::string fileName = spf(fileNameFormat,i);
    kernels[i] = a3read<float>(fileName);
    if (kernels[i].empty()) { printf("error loading %s\n",fileName.c_str()); return std::vector<A3f>(); }
  }
  return kernels;
}

inline VGG19 vgg19loadFromDir(const std::string& path)
{
  VGG19 vgg19;

  vgg19.conv1_1 = loadKernels(path+"/conv1_1/%03d.A3f",64);
  vgg19.conv1_2 = loadKernels(path+"/conv1_2/%03d.A3f",64);

  vgg19.conv2_1 = loadKernels(path+"/conv2_1/%03d.A3f",128);
  vgg19.conv2_2 = loadKernels(path+"/conv2_2/%03d.A3f",128);

  vgg19.conv3_1 = loadKernels(path+"/conv3_1/%03d.A3f",256);
  vgg19.conv3_2 = loadKernels(path+"/conv3_2/%03d.A3f",256);
  vgg19.conv3_3 = loadKernels(path+"/conv3_3/%03d.A3f",256);
  vgg19.conv3_4 = loadKernels(path+"/conv3_4/%03d.A3f",256);

  vgg19.conv4_1 = loadKernels(path+"/conv4_1/%03d.A3f",512);
  vgg19.conv4_2 = loadKernels(path+"/conv4_2/%03d.A3f",512);
  vgg19.conv4_3 = loadKernels(path+"/conv4_3/%03d.A3f",512);
  vgg19.conv4_4 = loadKernels(path+"/conv4_4/%03d.A3f",512);

  vgg19.conv5_1 = loadKernels(path+"/conv5_1/%03d.A3f",512);
  vgg19.conv5_2 = loadKernels(path+"/conv5_2/%03d.A3f",512);
  vgg19.conv5_3 = loadKernels(path+"/conv5_3/%03d.A3f",512);
  vgg19.conv5_4 = loadKernels(path+"/conv5_4/%03d.A3f",512);

  vgg19.conv5_4 = loadKernels(path+"/conv5_4/%03d.A3f",512);

  vgg19.fc6 = a2read<float>(path+"/fc6.A2f"); if (vgg19.fc6.empty()) { printf("error loading %s/fc6.A2f\n",path.c_str()); }
  vgg19.fc7 = a2read<float>(path+"/fc7.A2f"); if (vgg19.fc7.empty()) { printf("error loading %s/fc7.A2f\n",path.c_str()); }
  vgg19.fc8 = a2read<float>(path+"/fc8.A2f"); if (vgg19.fc8.empty()) { printf("error loading %s/fc8.A2f\n",path.c_str()); }

  return vgg19;
}

#endif
