SetFactory("OpenCASCADE");

Function Cuboid
l12 = newl; Line(l12) = {pnt[0], pnt[1]};
l23 = newl; Line(l23) = {pnt[1], pnt[2]};
l34 = newl; Line(l34) = {pnt[2], pnt[3]};
l41 = newl; Line(l41) = {pnt[3], pnt[0]};
l56 = newl; Line(l56) = {pnt[4], pnt[5]};
l67 = newl; Line(l67) = {pnt[5], pnt[6]};
l78 = newl; Line(l78) = {pnt[6], pnt[7]};
l85 = newl; Line(l85) = {pnt[7], pnt[4]};
l15 = newl; Line(l15) = {pnt[0], pnt[4]};
l26 = newl; Line(l26) = {pnt[1], pnt[5]};
l37 = newl; Line(l37) = {pnt[2], pnt[6]};
l48 = newl; Line(l48) = {pnt[3], pnt[7]};

ll1 = newll; Line Loop(ll1) = { l12, l23, l34, l41 };
ll2 = newll; Line Loop(ll2) = { l56, l67, l78, l85 };
ll3 = newll; Line Loop(ll3) = { l12, l26, -l56, -l15 };
ll4 = newll; Line Loop(ll4) = { l23, l37, -l67, -l26 };
ll5 = newll; Line Loop(ll5) = { l34, l48, -l78, -l37 };
ll6 = newll; Line Loop(ll6) = { l41, l15, -l85, -l48 };

s1 = news; Plane Surface(s1) = { ll1 } ;
s2 = news; Plane Surface(s2) = { ll2 } ;
s3 = news; Plane Surface(s3) = { ll3 } ;
s4 = news; Plane Surface(s4) = { ll4 } ;
s5 = news; Plane Surface(s5) = { ll5 } ;
s6 = news; Plane Surface(s6) = { ll6 } ;

sl = newsl; Surface Loop(sl) = { s1, s2, s3, s4, s5, s6 };
v = newv; Volume(v) = { sl };
Printf("Cuboid v=%g", v);

If( Flag_TransfInf )
  //Mesh.Algorithm3D = 4; 
  // (4=Frontal, 5=Frontal Delaunay, 6=Frontal Hex, 7=MMG3D, 9=R-tree)
  For num In { l12:l85 } 
    Transfinite Line{ num } = 10;
  EndFor
  For num In { l15:l48 } 
    Transfinite Line{ num } = 5;
  EndFor
  For num In { s1:s6 } 
    Transfinite Surface{ num } ;
  EndFor
  Transfinite Volume{ v } ;
EndIf
Return


DefineConstant[
  Flag_InfiniteBox = {1, Choices{0,1}, Name "Infinite box/Add infinite box"}
  Flag_TransfInf = {0, Choices{0,1}, Name "Infinite box/Transfinite mesh", Visible 0}
  ratioInf = {2, Name "Infinite box/Ratio ext-int", Visible Flag_InfiniteBox}
  ratioBox = {1.25, Name "Infinite box/Ratio int-content", Visible Flag_InfiniteBox}
  ratioLc = {10, Name "Infinite box/Ratio int-Lc", Visible Flag_InfiniteBox}

  xInt = {1, Name "Infinite box/xInt", Visible 0}
  yInt = {1, Name "Infinite box/yInt", Visible 0}
  zInt = {1, Name "Infinite box/zInt", Visible 0}
  xExt = {xInt*ratioInf, Name "Infinite box/xExt", Visible 0}
  yExt = {yInt*ratioInf, Name "Infinite box/yExt", Visible 0}
  zExt = {zInt*ratioInf, Name "Infinite box/zExt", Visible 0}
  xCnt = {0, Name "Infinite box/xCenter", Visible 0}
  yCnt = {0, Name "Infinite box/yCenter", Visible 0}
  zCnt = {0, Name "Infinite box/zCenter", Visible 0}
];

// Compute parameters related to the Infinite box
BoundingBox;
xCnt = (General.MinX + General.MaxX) / 2.;
yCnt = (General.MinY + General.MaxY) / 2.;
zCnt = (General.MinZ + General.MaxZ) / 2.;

xInt = (General.MaxX - General.MinX)/2.;
yInt = (General.MaxY - General.MinZ)/2.;
zInt = (General.MaxZ - General.MinZ)/2.;
// If BoundingBox is empty, replace it with a unit cube.
// This makes this file callable in isolation, e.g., for testing
If (xInt == 0 )
  xInt=1;
EndIf
If (yInt == 0 )
  yInt=1;
EndIf
If (zInt == 0 )
  zInt=1;
EndIf

// Compute aspect ratio of the Infinite Box
diagInt = Sqrt[ xInt^2 + yInt^2 + zInt^2 ];
pp = 1.2; // pp=1 square InfBox, pp>3 Box and content have the same aspect ratio
xInt *= ratioBox*(diagInt/xInt)^(1/pp); 
yInt *= ratioBox*(diagInt/yInt)^(1/pp); 
zInt *= ratioBox*(diagInt/zInt)^(1/pp); 


// FIXME is there another (better) way to convey the values calculated here to getDP
SetNumber("Infinite box/xInt",xInt);
SetNumber("Infinite box/yInt",yInt);
SetNumber("Infinite box/zInt",zInt);
SetNumber("Infinite box/xExt",xExt);
SetNumber("Infinite box/yExt",yExt);
SetNumber("Infinite box/zExt",zExt);
SetNumber("Infinite box/xCenter",xCnt);
SetNumber("Infinite box/yCenter",yCnt);
SetNumber("Infinite box/zCenter",zCnt);

lc1inf = Sqrt[ xInt^2 + yInt^2 + zInt^2] / ratioLc;
p1 = newp; Point (p1) = {xCnt-xInt, yCnt-yInt, zCnt-zInt, lc1inf};
p2 = newp; Point (p2) = {xCnt+xInt, yCnt-yInt, zCnt-zInt, lc1inf};
p3 = newp; Point (p3) = {xCnt+xInt, yCnt+yInt, zCnt-zInt, lc1inf};
p4 = newp; Point (p4) = {xCnt-xInt, yCnt+yInt, zCnt-zInt, lc1inf};
p5 = newp; Point (p5) = {xCnt-xInt, yCnt-yInt, zCnt+zInt, lc1inf};
p6 = newp; Point (p6) = {xCnt+xInt, yCnt-yInt, zCnt+zInt, lc1inf};
p7 = newp; Point (p7) = {xCnt+xInt, yCnt+yInt, zCnt+zInt, lc1inf};
p8 = newp; Point (p8) = {xCnt-xInt, yCnt+yInt, zCnt+zInt, lc1inf};

VolInf[]={}; // Define empty array in case Flag_InfiniteBox is not active
If(Flag_InfiniteBox)
  xExt = xInt * ratioInf;
  yExt = yInt * ratioInf;
  zExt = zInt * ratioInf;

  lc2inf = lc1inf;
  pp1 = newp; Point (pp1) = {xCnt-xExt, yCnt-yExt, zCnt-zExt, lc2inf};
  pp2 = newp; Point (pp2) = {xCnt+xExt, yCnt-yExt, zCnt-zExt, lc2inf};
  pp3 = newp; Point (pp3) = {xCnt+xExt, yCnt+yExt, zCnt-zExt, lc2inf};
  pp4 = newp; Point (pp4) = {xCnt-xExt, yCnt+yExt, zCnt-zExt, lc2inf};
  pp5 = newp; Point (pp5) = {xCnt-xExt, yCnt-yExt, zCnt+zExt, lc2inf};
  pp6 = newp; Point (pp6) = {xCnt+xExt, yCnt-yExt, zCnt+zExt, lc2inf};
  pp7 = newp; Point (pp7) = {xCnt+xExt, yCnt+yExt, zCnt+zExt, lc2inf};
  pp8 = newp; Point (pp8) = {xCnt-xExt, yCnt+yExt, zCnt+zExt, lc2inf};

  pnt[]={p1,p2,p3,p4,pp1,pp2,pp3,pp4}; Call Cuboid; 
  VolInf[] = v;
  // FIXME seems to be forbidden to have the command "VolInf[] = v;" 
  // on the same line as "Call Cuboid" 
  pnt[]={p5,p6,p7,p8,pp5,pp6,pp7,pp8}; Call Cuboid; 
  VolInf[] += v;
  pnt[]={p1,p2,p6,p5,pp1,pp2,pp6,pp5}; Call Cuboid; 
  VolInf[] += {v};
  pnt[]={p3,p4,p8,p7,pp3,pp4,pp8,pp7}; Call Cuboid; 
  VolInf[] += {v};
  pnt[]={p2,p3,p7,p6,pp2,pp3,pp7,pp6}; Call Cuboid; 
  VolInf[] += {v};
  pnt[]={p4,p1,p5,p8,pp4,pp1,pp5,pp8}; Call Cuboid; 
  VolInf[] += {v};

  Physical Volume("InfiniteX", 1) = { VolInf[4], VolInf[5] };
  Physical Volume("InfiniteY", 2) = { VolInf[2], VolInf[3] };
  Physical Volume("InfiniteZ", 3) = { VolInf[0], VolInf[1] };
EndIf

pnt[]={p1,p2,p3,p4,p5,p6,p7,p8}; Call Cuboid;
InteriorInfBox = v;

For num In {0:#VolInf()-1}
   Printf("VolInf %5g", VolInf[num]);
EndFor






