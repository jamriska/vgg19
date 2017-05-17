#include <cstdio>

#include "jzq.h"
#include "imageio.h"
#include "resample.h"

#include "vgg19.h"

#define FOR(A,X,Y) for(int Y=0;Y<A.height();Y++) for(int X=0;X<A.width();X++)

template<typename T>
Array2<T> hstack(const Array2<T>& A,const Array2<T>& B)
{
  Array2<T> AB(A.width()+B.width(),std::max(A.height(),B.height()));
  fill(&AB,zero<T>::value());
  FOR(A,x,y) { AB(x,y) = A(x,y); }
  FOR(B,x,y) { AB(x+A.width(),y) = B(x,y); }
  return AB;
}

template<typename T>
Array2<T> vstack(const Array2<T>& A,const Array2<T>& B)
{
  Array2<T> AB(std::max(A.width(),B.width()),A.height()+B.height());
  fill(&AB,V3f(0.5,0.5,0.5));
  FOR(A,x,y) { AB(x,y) = A(x,y); }
  FOR(B,x,y) { AB(x,y+A.height()) = B(x,y); }
  return AB;
}

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

A2V3f mosaic(const A3f& X,int width)
{
  const int count = X.depth();
  const int numCols = std::ceil(std::sqrt(float(count)));
  const int numRows = std::ceil(float(count)/float(numCols));

  const A3f normX = normalize(X);

  A2V3f O;
  for(int row=0;row<numRows;row++)
  {
    A2V3f R;
    for(int col=0;col<numCols;col++)
    {
      const int index = col+row*numCols;
      if (index<count)
      {
        A2f S = slice(normX,index);
        A2V3f I(size(S));
        for(int xy=0;xy<numel(I);xy++)
        {
          const float s = std::min(1.5f*std::pow(S[xy],1.0f/1.2f),1.0f);
          I[xy] = V3f(s,s,s);
        }
        const float aspect = float(I.height())/float(I.width());
        I = resample(I,V2i(width,aspect*float(width)),filterBox);
        A2V3f I2(size(I)+V2i(1,1));
        fill(&I2,V3f(0.5,0.5,0.5));
        FOR(I,x,y) { I2(x,y)=I(x,y); }
        R = (col==0) ? I2 : hstack(R,I2);
      }
    }
    O = (row==0) ? R : vstack(O,R);
  }

  return O;
}

int main(int argc,char** argv)
{
  if (argc<2)
  {
    printf("usage: featmosaic.exe image.png [output_%%d.png]\n");
    return 1;
  }

  const A2V3f image = imread<V3f>(argv[1]);

  if (image.empty()) { printf("failed to read %s\n",argv[1]); return 1; }

  const VGG19 vgg19 = vgg19loadFromDir("./vgg19");

  for(int layer=1;layer<=5;layer++)
  {
    const A3f X = vgg19forwardPass(image,vgg19,layer);

    A2V3f O;

    if      (layer==1) { O = mosaic(X,105); }
    else if (layer==2) { O = mosaic(X,70);  }
    else if (layer==3) { O = mosaic(X,52);  }
    else if (layer==4) { O = mosaic(X,36);  }
    else if (layer==5) { O = mosaic(X,36);  }

    imwrite(O,spf(argc>2 ? argv[2] : "L%d.png",layer));
  }

  return 0;
}
