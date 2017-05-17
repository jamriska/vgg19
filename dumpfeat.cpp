#include <cstdio>

#include "jzq.h"
#include "imageio.h"

#include "vgg19.h"

A3f normalize(const A3f& input)
{
  A2f length(input.width(),input.height());

  for(int y=0;y<input.height();y++)
  for(int x=0;x<input.width();x++)
  {
    float sum = 0;
    for(int z=0;z<input.depth();z++) { sum += input(x,y,z)*input(x,y,z); }
    length(x,y) = std::sqrt(sum);
  }

  A3f output(size(input));
  for(int z=0;z<input.depth();z++)
  for(int y=0;y<input.height();y++)
  for(int x=0;x<input.width();x++)
  {
    output(x,y,z) = input(x,y,z)/length(x,y);
  }

  const float mn = min(output);
  const float mx = max(output);

  for(int xyz=0;xyz<numel(output);xyz++)
  {
    output[xyz] = (output[xyz]-mn)/(mx-mn);
  }

  return output;
}

A2f slice(const A3f& input,const int z)
{
  A2f O(input.width(),input.height());

  for(int y=0;y<input.height();y++)
  for(int x=0;x<input.width();x++)
  {
    O(x,y) = input(x,y,z);
  }

  return O;
}

int main(int argc,char** argv)
{
  if (argc<2)
  {
    printf("usage: dumpfeat.exe image.png [layer] [output_%%d.png]\n");
    return 1;
  }

  const A2V3f image = imread<V3f>(argv[1]);

  if (image.empty()) { printf("failed to read %s\n",argv[1]); return 1; }

  const VGG19 vgg19 = vgg19loadFromDir("./vgg19");

  const int layer = argc>2 ? atoi(argv[2]) : 5;

  const A3f X = vgg19forwardPass(image,vgg19,layer);

  const A3f normX = normalize(X);

  for(int z=0;z<normX.depth();z++)
  {
    imwrite(slice(normX,z),spf(argc>3 ? argv[3] : "%03d.png",z));
  }

  return 0;
}
