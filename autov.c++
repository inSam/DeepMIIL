#include <iostream>
#include <stdio.h>
#include <sstream> 
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Scalar getMSSIM( const Mat& i1, const Mat& i2);

int main(){
    string dirname;
    DIR *dir;
    struct dirent *ent;
        
    do{
        cout << "Please input directory for trained output: " << endl;
        getline(cin, dirname);
        
    }while(!(dir = opendir(dirname.c_str())));


    //Construct and sort vector of output and target files
    vector<string> outputList;
    vector<string> targetList;
    while((ent = readdir(dir)) != NULL){
        string name = ent->d_name;
        if(name.find("outputs") != string::npos) {
            outputList.push_back(dirname + "/" + ent->d_name);
        }else if(name.find("targets") != string::npos){
            targetList.push_back(dirname + "/" + ent->d_name);
        }
    }

    sort(outputList.begin(), outputList.end());
    sort(targetList.begin(), targetList.end());

    
    vector<double> MSE;
    vector<Scalar> SSIM;
    //callculates MSE; 
       for(size_t i = 0; i < outputList.size(); i++){
        cv::Mat output = cv::imread(outputList[i]);
        cv::Mat input = cv::imread(targetList[i]);
        cv::Mat diff;
        
        cv::absdiff(output.mul(output), input.mul(input), diff);
        cv::Scalar sum = cv::sum(diff);
        MSE.push_back(sum.val[0] + sum.val[1] + sum.val[2]);

        SSIM.push_back(getMSSIM(output, input));
    }

    for(size_t i = 0; i < outputList.size(); i++){
        cout << outputList[i] << endl << " MSE: " << MSE[i] << endl;
        cout << " MSSIM: "
             << " R " << setiosflags(ios::fixed) << setprecision(2) << SSIM[i].val[2] * 100 << "%"
             << " G " << setiosflags(ios::fixed) << setprecision(2) << SSIM[i].val[1] * 100 << "%"
             << " B " << setiosflags(ios::fixed) << setprecision(2) << SSIM[i].val[0] * 100 << "%" << endl;
    }
}


double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);        // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;

    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}
