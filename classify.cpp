#include <cstdio>

#include "jzq.h"
#include "imageio.h"
#include "resample.h"

#include "vgg19.h"
#include "classlabels.h"

A2V3f resizeTo224x224(const A2V3f& I)
{
  A2V3f O(max(size(I)),max(size(I)));

  for(int y=0;y<O.height();y++)
  for(int x=0;x<O.width();x++)
  {
    O(x,y) = I(clamp(V2i(x,y)-(size(O)-size(I))/2,
                     V2i(0,0),size(I)-V2i(1,1)));
  }

  O = resample(O,V2i(224,224),filterBox);

  return O;
}

int main(int argc,char** argv)
{
  if (argc!=2)
  {
    printf("usage: classify.exe image.png\n");
    return 1;
  }

  const A2V3f image = imread<V3f>(argv[1]);

  if (image.empty()) { printf("failed to read %s\n",argv[1]); return 1; }

  const VGG19 vgg19 = vgg19loadFromDir("./vgg19");

  const A3f X = vgg19forwardPass(resizeTo224x224(image),vgg19);

  const char* label = classLabels[argmax(X)[2]];

  printf("result: %s\n",label);

  return 0;
}
