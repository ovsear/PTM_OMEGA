/*Copyright (c) 2016 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef PTM_QUAT_H
#define PTM_QUAT_H

namespace ptm {

const double generator_cubic[24][4] = {
        {          1,          0,          0,          0 },
        {  sqrt(2)/2,  sqrt(2)/2,          0,          0 },
        {  sqrt(2)/2,          0,  sqrt(2)/2,          0 },
        {  sqrt(2)/2,          0,          0,  sqrt(2)/2 },
        {  sqrt(2)/2,          0,          0, -sqrt(2)/2 },
        {  sqrt(2)/2,          0, -sqrt(2)/2,          0 },
        {  sqrt(2)/2, -sqrt(2)/2,         -0,         -0 },
        {        0.5,        0.5,        0.5,        0.5 },
        {        0.5,        0.5,        0.5,       -0.5 },
        {        0.5,        0.5,       -0.5,        0.5 },
        {        0.5,        0.5,       -0.5,       -0.5 },
        {        0.5,       -0.5,        0.5,        0.5 },
        {        0.5,       -0.5,        0.5,       -0.5 },
        {        0.5,       -0.5,       -0.5,        0.5 },
        {        0.5,       -0.5,       -0.5,       -0.5 },
        {          0,          1,          0,          0 },
        {          0,  sqrt(2)/2,  sqrt(2)/2,          0 },
        {          0,  sqrt(2)/2,          0,  sqrt(2)/2 },
        {          0,  sqrt(2)/2,          0, -sqrt(2)/2 },
        {          0,  sqrt(2)/2, -sqrt(2)/2,          0 },
        {          0,          0,          1,          0 },
        {          0,          0,  sqrt(2)/2,  sqrt(2)/2 },
        {          0,          0,  sqrt(2)/2, -sqrt(2)/2 },
        {          0,          0,          0,          1 },
};

const double generator_diamond_cubic[12][4] = {
        {    1,    0,    0,    0 },
        {  0.5,  0.5,  0.5,  0.5 },
        {  0.5,  0.5,  0.5, -0.5 },
        {  0.5,  0.5, -0.5,  0.5 },
        {  0.5,  0.5, -0.5, -0.5 },
        {  0.5, -0.5,  0.5,  0.5 },
        {  0.5, -0.5,  0.5, -0.5 },
        {  0.5, -0.5, -0.5,  0.5 },
        {  0.5, -0.5, -0.5, -0.5 },
        {    0,    1,    0,    0 },
        {    0,    0,    1,    0 },
        {    0,    0,    0,    1 },
};

const double generator_hcp[6][4] = {
        {          1,          0,          0,          0 },
        {        0.5,          0,          0,  sqrt(3)/2 },
        {        0.5,          0,          0, -sqrt(3)/2 },
        {          0,  sqrt(3)/2,        0.5,          0 },
        {          0,  sqrt(3)/2,       -0.5,          0 },
        {          0,          0,          1,          0 },
};


const double generator_hcp_conventional[12][4] = {
        {          1,          0,          0,          0 },
        {  sqrt(3)/2,          0,          0,        0.5 },
        {  sqrt(3)/2,          0,          0,       -0.5 },
        {        0.5,          0,          0,  sqrt(3)/2 },
        {        0.5,          0,          0, -sqrt(3)/2 },
        {          0,          1,          0,          0 },
        {          0,  sqrt(3)/2,        0.5,          0 },
        {          0,  sqrt(3)/2,       -0.5,          0 },
        {          0,        0.5,  sqrt(3)/2,          0 },
        {          0,        0.5, -sqrt(3)/2,          0 },
        {          0,          0,          1,          0 },
        {          0,          0,          0,          1 },
};

const double generator_diamond_hexagonal[3][4] = {
        {          1,          0,          0,          0 },
        {        0.5,          0,          0,  sqrt(3)/2 },
        {        0.5,          0,          0, -sqrt(3)/2 },
};

const double generator_icosahedral[60][4] = {
        {                        1,                        0,                        0,                        0 },
        {            (1+sqrt(5))/4,                      0.5,   sqrt(25-10*sqrt(5))/10,   sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,                      0.5,  -sqrt(25-10*sqrt(5))/10,  -sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,            1/(1+sqrt(5)),   sqrt(10*sqrt(5)+50)/20,  -sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,            1/(1+sqrt(5)),  -sqrt(10*sqrt(5)+50)/20,   sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,                        0,   sqrt(50-10*sqrt(5))/10,   sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,                        0,                        0,     sqrt(5./8-sqrt(5)/8) },
        {            (1+sqrt(5))/4,                        0,                        0,    -sqrt(5./8-sqrt(5)/8) },
        {            (1+sqrt(5))/4,                        0,  -sqrt(50-10*sqrt(5))/10,  -sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,           -1/(1+sqrt(5)),   sqrt(10*sqrt(5)+50)/20,  -sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,           -1/(1+sqrt(5)),  -sqrt(10*sqrt(5)+50)/20,   sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,                     -0.5,   sqrt(25-10*sqrt(5))/10,   sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,                     -0.5,  -sqrt(25-10*sqrt(5))/10,  -sqrt(50-10*sqrt(5))/20 },
        {                      0.5,            (1+sqrt(5))/4,   sqrt(50-10*sqrt(5))/20,  -sqrt(25-10*sqrt(5))/10 },
        {                      0.5,            (1+sqrt(5))/4,  -sqrt(50-10*sqrt(5))/20,   sqrt(25-10*sqrt(5))/10 },
        {                      0.5,                      0.5,   sqrt((5+2*sqrt(5))/20),   sqrt(25-10*sqrt(5))/10 },
        {                      0.5,                      0.5,   sqrt(25-10*sqrt(5))/10,  -sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                      0.5,  -sqrt(25-10*sqrt(5))/10,   sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                      0.5,  -sqrt((5+2*sqrt(5))/20),  -sqrt(25-10*sqrt(5))/10 },
        {                      0.5,            1/(1+sqrt(5)),   sqrt(10*sqrt(5)+50)/20,   sqrt((5+2*sqrt(5))/20) },
        {                      0.5,            1/(1+sqrt(5)),  -sqrt(10*sqrt(5)+50)/20,  -sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                        0,     sqrt((5+sqrt(5))/10),  -sqrt(25-10*sqrt(5))/10 },
        {                      0.5,                        0,   sqrt(50-10*sqrt(5))/10,  -sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                        0,  -sqrt(50-10*sqrt(5))/10,   sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                        0,    -sqrt((5+sqrt(5))/10),   sqrt(25-10*sqrt(5))/10 },
        {                      0.5,           -1/(1+sqrt(5)),   sqrt(10*sqrt(5)+50)/20,   sqrt((5+2*sqrt(5))/20) },
        {                      0.5,           -1/(1+sqrt(5)),  -sqrt(10*sqrt(5)+50)/20,  -sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                     -0.5,   sqrt((5+2*sqrt(5))/20),   sqrt(25-10*sqrt(5))/10 },
        {                      0.5,                     -0.5,   sqrt(25-10*sqrt(5))/10,  -sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                     -0.5,  -sqrt(25-10*sqrt(5))/10,   sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                     -0.5,  -sqrt((5+2*sqrt(5))/20),  -sqrt(25-10*sqrt(5))/10 },
        {                      0.5,           -(1+sqrt(5))/4,   sqrt(50-10*sqrt(5))/20,  -sqrt(25-10*sqrt(5))/10 },
        {                      0.5,           -(1+sqrt(5))/4,  -sqrt(50-10*sqrt(5))/20,   sqrt(25-10*sqrt(5))/10 },
        {            1/(1+sqrt(5)),            (1+sqrt(5))/4,   sqrt(50-10*sqrt(5))/20,   sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),            (1+sqrt(5))/4,  -sqrt(50-10*sqrt(5))/20,  -sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                      0.5,   sqrt((5+2*sqrt(5))/20),  -sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                      0.5,  -sqrt((5+2*sqrt(5))/20),   sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                        0,     sqrt((5+sqrt(5))/10),   sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                        0,                        0,  sqrt(1-1/(2*sqrt(5)+6)) },
        {            1/(1+sqrt(5)),                        0,                        0, -sqrt(1-1/(2*sqrt(5)+6)) },
        {            1/(1+sqrt(5)),                        0,    -sqrt((5+sqrt(5))/10),  -sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                     -0.5,   sqrt((5+2*sqrt(5))/20),  -sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                     -0.5,  -sqrt((5+2*sqrt(5))/20),   sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),           -(1+sqrt(5))/4,   sqrt(50-10*sqrt(5))/20,   sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),           -(1+sqrt(5))/4,  -sqrt(50-10*sqrt(5))/20,  -sqrt(10*sqrt(5)+50)/20 },
        {                        0,                        1,                        0,                        0 },
        {                        0,            (1+sqrt(5))/4,     sqrt(5./8-sqrt(5)/8),                        0 },
        {                        0,            (1+sqrt(5))/4,   sqrt(50-10*sqrt(5))/20,  -sqrt(50-10*sqrt(5))/10 },
        {                        0,            (1+sqrt(5))/4,  -sqrt(50-10*sqrt(5))/20,   sqrt(50-10*sqrt(5))/10 },
        {                        0,            (1+sqrt(5))/4,    -sqrt(5./8-sqrt(5)/8),                        0 },
        {                        0,                      0.5,   sqrt((5+2*sqrt(5))/20),   sqrt(50-10*sqrt(5))/10 },
        {                        0,                      0.5,   sqrt(25-10*sqrt(5))/10,     sqrt((5+sqrt(5))/10) },
        {                        0,                      0.5,  -sqrt(25-10*sqrt(5))/10,    -sqrt((5+sqrt(5))/10) },
        {                        0,                      0.5,  -sqrt((5+2*sqrt(5))/20),  -sqrt(50-10*sqrt(5))/10 },
        {                        0,            1/(1+sqrt(5)),  sqrt(1-1/(2*sqrt(5)+6)),                        0 },
        {                        0,            1/(1+sqrt(5)),   sqrt(10*sqrt(5)+50)/20,    -sqrt((5+sqrt(5))/10) },
        {                        0,            1/(1+sqrt(5)),  -sqrt(10*sqrt(5)+50)/20,     sqrt((5+sqrt(5))/10) },
        {                        0,            1/(1+sqrt(5)), -sqrt(1-1/(2*sqrt(5)+6)),                        0 },
        {                        0,                        0,     sqrt((5+sqrt(5))/10),  -sqrt(50-10*sqrt(5))/10 },
        {                        0,                        0,   sqrt(50-10*sqrt(5))/10,     sqrt((5+sqrt(5))/10) },
};

const double generator_omega_a[12][4] = {
        { 1, 0, 0, 0 }, // C6^0
        { sqrt(3)/2, 0, 0, 0.5 }, // C6^1
        { 0.5, 0, 0, sqrt(3)/2 }, // C6^2
        { 0, 0, 0, 1 }, // C6^3
        { 0.5, 0, 0, -sqrt(3)/2 }, // C6^4
        { sqrt(3)/2, 0, 0, -0.5 }, // C6^5
        { 0, 1, 0, 0 }, // C2_axis_0deg
        { 0, sqrt(3)/2, 0.5, 0 }, // C2_axis_30deg
        { 0, 0.5, sqrt(3)/2, 0 }, // C2_axis_60deg
        { 0, 0, 1, 0 }, // C2_axis_90deg
        { 0, -0.5, sqrt(3)/2, 0 }, // C2_axis_120deg
        { 0, sqrt(3)/2, -0.5, 0 }, // C2_axis_150deg
};

const double generator_omega_b[6][4] = {
        { 1, 0, 0, 0 }, // C3_like_0deg
        { 0.5, 0, 0, sqrt(3)/2 }, // C3_like_120deg
        { 0.5, 0, 0, -sqrt(3)/2 }, // C3_like_240deg
        { 0, sqrt(3)/2, 0.5, 0 }, // C2_axis_30deg
        { 0, 0, 1, 0 }, // C2_axis_90deg
        { 0, sqrt(3)/2, -0.5, 0 }, // C2_axis_150deg
};

const double generator_omega_b_conventional[12][4] = {
        {          1,          0,          0,          0 },
        {  sqrt(3)/2,          0,          0,        0.5 },
        {  sqrt(3)/2,          0,          0,       -0.5 },
        {        0.5,          0,          0,  sqrt(3)/2 },
        {        0.5,          0,          0, -sqrt(3)/2 },

        {          0,          1,          0,          0 },
        {          0,  sqrt(3)/2,        0.5,          0 },
        {          0,  sqrt(3)/2,       -0.5,          0 },
        {          0,        0.5,  sqrt(3)/2,          0 },
        {          0,        0.5, -sqrt(3)/2,          0 },
        {          0,          0,          1,          0 },
        {          0,          0,          0,          1 },
};


int rotate_quaternion_into_cubic_fundamental_zone(double* q);
int rotate_quaternion_into_diamond_cubic_fundamental_zone(double* q);
int rotate_quaternion_into_icosahedral_fundamental_zone(double* q);
int rotate_quaternion_into_hcp_fundamental_zone(double* q);
int rotate_quaternion_into_hcp_conventional_fundamental_zone(double* q);
int rotate_quaternion_into_diamond_hexagonal_fundamental_zone(double* q);
int rotate_quaternion_into_omega_a_fundamental_zone(double* q);
int rotate_quaternion_into_omega_b_fundamental_zone(double* q);
int rotate_quaternion_into_omega_b_conventional_fundamental_zone(double* q);

void quat_rot(double* r, double* a, double* b);
void normalize_quaternion(double* q);
void quaternion_to_rotation_matrix(double* q, double* U);
void rotation_matrix_to_quaternion(double* u, double* q);
double quat_dot(double* a, double* b);
double quat_misorientation(double* q1, double* q2);

int map_quaternion_cubic(double* q, int i);
int map_quaternion_diamond_cubic(double* q, int i);
int map_quaternion_icosahedral(double* q, int i);
int map_quaternion_hcp(double* q, int i);
int map_quaternion_hcp_conventional(double* q, int i);
int map_quaternion_diamond_hexagonal(double* q, int i);
int map_quaternion_omega_a(double* q, int i);
int map_quaternion_omega_b(double* q, int i);
int map_quaternion_omega_b_conventional(double* q, int i);

}

#endif

