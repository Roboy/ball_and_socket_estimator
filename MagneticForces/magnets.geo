SetFactory("OpenCASCADE");

Include "magnets_common.pro";

mm = 1.e-3;

SetFactory("OpenCASCADE");


// set some global Gmsh options
Mesh.Optimize = 1; // optimize quality of tetrahedra
Mesh.VolumeEdges = 0; // Toggle mesh display
Mesh.SurfaceEdges = 0;
Solver.AutoMesh = 2; // always remesh if necessary (don't reuse mesh on disk)

VolMagnets[] = {};
For i In {1:NumMagnets}
  iLabel = newv;
  If(M~{i} == 0) // cylinder
    Cylinder(iLabel) = { X~{i}, Y~{i}-L~{i}/2, Z~{i}, 0, L~{i}, 0, R~{i} };
  ElseIf(M~{i} == 1) // parallelepiped
    Box(iLabel) = { X~{i}-Lx~{i}/2, Y~{i}-Ly~{i}/2, Z~{i}-Lz~{i}/2, Lx~{i}, Ly~{i}, Lz~{i} };
  EndIf

  Rotate { {0,0,1}, {X~{i},Y~{i},Z~{i}}, deg*Rz~{i} } { Volume{ iLabel }; }
  Rotate { {0,1,0}, {X~{i},Y~{i},Z~{i}}, deg*Ry~{i} } { Volume{ iLabel }; }
  Rotate { {1,0,0}, {X~{i},Y~{i},Z~{i}}, deg*Rx~{i} } { Volume{ iLabel }; }

  Physical Volume(Sprintf("Magnet_%g",i),10*i) = { iLabel };
  skin~{i}[] = CombinedBoundary{ Volume{ iLabel }; };
  Physical Surface(Sprintf("SkinMagnet_%g",i),10*i+1) = -skin~{i}[]; // magnet skin
  VolMagnets[] += { iLabel };
EndFor

VolSensors[] = {};
For i In {1:NumSensors}
  iLabel = newv;

  Box(iLabel) = { X~{i+NumMagnets}-Lx~{i+NumMagnets}/2, Y~{i+NumMagnets}-Ly~{i+NumMagnets}/2, Z~{i+NumMagnets}-Lz~{i+NumMagnets}/2, Lx~{i+NumMagnets}, Ly~{i+NumMagnets}, Lz~{i+NumMagnets} };

  Rotate { {0,0,1}, {X~{i+NumMagnets},Y~{i+NumMagnets},Z~{i+NumMagnets}}, deg*Rz~{i+NumMagnets} } { Volume{ iLabel }; }
  Rotate { {0,1,0}, {X~{i+NumMagnets},Y~{i+NumMagnets},Z~{i+NumMagnets}}, deg*Ry~{i+NumMagnets} } { Volume{ iLabel }; }
  Rotate { {1,0,0}, {X~{i+NumMagnets},Y~{i+NumMagnets},Z~{i+NumMagnets}}, deg*Rx~{i+NumMagnets} } { Volume{ iLabel }; }

  Physical Volume(Sprintf("Sensor_%g",i),10*i+10*NumMagnets) = { iLabel };
  skin~{i}[] = CombinedBoundary{ Volume{ iLabel }; };
  Physical Surface(Sprintf("SkinSensor_%g",i),10*i+1+10*NumMagnets) = -skin~{i}[]; // sensor skin
  VolSensors[] += { iLabel };
EndFor


Include "InfiniteBox.geo";

// The overall dimensions of the model have been calculated in InfiniteBox.geo
// So we use to characteristic length set for the infinite box for the whole mesh.
Mesh.CharacteristicLengthMin = lc1inf;
Mesh.CharacteristicLengthMax = lc1inf;


AirBox[] = BooleanDifference{
  Volume{ InteriorInfBox }; Delete;
}{
  Volume{ VolMagnets[],VolSensors[] };
};

Physical Volume("AirBox",4) = { AirBox[] };

Volumes[] = { VolInf[], VolMagnets[], VolSensors[], AirBox[] };

vv[] = BooleanFragments{
  Volume { Volumes[] }; Delete; }{};

If( #vv[] > #Volumes[] )
  Error("Overlapping magnets");
  Abort;
EndIf

Printf("Check whether BooleanFragments has preserved volume numbering:");
For num In {0:#vv()-1}
   Printf("Fragment %5g -> %g", Volumes[num], vv(num));
EndFor

Outer[] = CombinedBoundary{ Volume{ Volumes[] }; };
Physical Surface("OuterSurface", 5) = { Outer[]  };
