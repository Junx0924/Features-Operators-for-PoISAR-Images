#pragma once
#ifndef  PARAM_H_
#define  PARAM_H_
#include <string>
#include <filesystem>
#include <iostream>
using namespace std;

string pol = "/data";
vector<string> pol_dataset = { "/hh" , "/vv" ,"/hv" ,"/vh"};

string ctElements = "/CTelememts";
string MP = "/MP";
string decomp = "/decomp";
string color = "/color";
string texture = "/texture";
string polStatistic = "/polStatistic";

vector<string> feature_type = { texture, color, ctElements,polStatistic,decomp, MP };
vector<string> dataset_name = { "/feature" ,"/patchLabel" };
string knn_result = { "/knn_result" };


#endif