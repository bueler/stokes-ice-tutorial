// hand-written trapezoid geometry

// describe geometry
lc = 5000.0;
Point(1) = {0.0,0.0,0,lc};
Point(2) = {20000.0,0.0,0,lc};
Point(3) = {15000.0,1000.0,0,lc};
Point(4) = {5000.0,1000.0,0,lc};
Line(5) = {1,2};
Line(6) = {2,3};
Line(7) = {3,4};
Line(8) = {4,1};
Line Loop(9) = {5,6,7,8};
Plane Surface(10) = {9};

// add labels so solver knows where to apply boundary conditions
//   boundary ids:  41, 42 = top, base
Physical Line(41) = {6,7,8};
Physical Line(42) = {5};
Physical Surface(51) = {10};   // if Physical Surface omitted, gmsh fails
