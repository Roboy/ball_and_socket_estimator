mm = 1.e-3;
deg = Pi/180.;
DefineConstant[
  NumMagnets = {3, Min 1, Max 20, Step 1, Name "Parameters/0Number of magnets"}
  NumSensors = {4, Min 1, Max 20, Step 1, Name "Parameters/0Number of sensors"}
  Flag_InfiniteBox = {1, Choices{0,1}, Name "Infinite box/Add infinite box"}
  Flag_FullMenu = {0, Choices{0,1}, Name "Parameters/Show all parameters"}
];

SensorPositionX_1 = -22.4;
SensorPositionX_2 = -14.0;
SensorPositionX_3 = 14.3;
SensorPositionX_4 = 22.325;

SensorPositionY_1 = 7.25;
SensorPositionY_2 = -19.5;
SensorPositionY_3 = -19.6;
SensorPositionY_4 = 6.675;

MagnetRotX_1 = -90;
MagnetRotY_1 = 30;
MagnetRotZ_1 = 0;

MagnetRotX_2 = 90;
MagnetRotY_2 = 30;
MagnetRotZ_2 = 0;

MagnetRotX_3 = 0;
MagnetRotY_3 = 0;
MagnetRotZ_3 = 180;

For i In {1:NumMagnets}
  DefineConstant[
    X~{i} = {12.5*Sin[Pi/180.0*120*i]*mm, Min -100*mm, Max 100*mm, Step mm, Visible 1,
      Name Sprintf("Parameters/Magnet %g/0X position [m]", i) },
    Y~{i} = {12.5*Cos[Pi/180.0*120*i]*mm, Min -100*mm, Max 100*mm, Step mm, Visible 1,
      Name Sprintf("Parameters/Magnet %g/0Y position [m]", i) },
    Z~{i} = {0, Min -100*mm, Max 100*mm, Step mm, Visible 1,
      Name Sprintf("Parameters/Magnet %g/0Z position [m]", i) },

    M~{i} = {1, Choices{0="Cylinder",1="Cube"},
      Name Sprintf("Parameters/Magnet %g/00Shape", i)},

    R~{i} = {20*mm, Min mm, Max 100*mm, Step mm,
      Name Sprintf("Parameters/Magnet %g/1Radius [m]", i),
      Visible (M~{i} == 0) },
    L~{i} = {50*mm, Min mm, Max 100*mm, Step mm,
      Name Sprintf("Parameters/Magnet %g/1Length [m]", i),
      Visible (M~{i} == 0) },

    Lx~{i} = {10*mm, Min mm, Max 100*mm, Step mm,
      Name Sprintf("Parameters/Magnet %g/1X length [m]", i),
      Visible (M~{i} == 1) },
    Ly~{i} = {10*mm, Min mm, Max 100*mm, Step mm,Visible Flag_FullMenu,
      Name Sprintf("Parameters/Magnet %g/1XY aspect ratio", i),
      Visible (M~{i} == 1) },
    Lz~{i} = {10*mm, Min mm, Max 100*mm, Step mm,Visible Flag_FullMenu,
      Name Sprintf("Parameters/Magnet %g/1XZ aspect ration", i),
      Visible (M~{i} == 1) },

    Rx~{i} = {MagnetRotX~{i}, Min -180, Max 180, Step 1,
      Name Sprintf("Parameters/Magnet %g/2X rotation [deg]", i) },
    Ry~{i} = {MagnetRotY~{i}, Min -180, Max 180, Step 1,
      Name Sprintf("Parameters/Magnet %g/2Y rotation [deg]", i) },
    Rz~{i} = {MagnetRotZ~{i}, Min -180, Max 180, Step 1,
      Name Sprintf("Parameters/Magnet %g/2Z rotation [deg]", i) },

    MUR~{i} = {5000,
      Name Sprintf("Parameters/Magnet %g/3Mu relative []", i)},
    BR~{i} = {1.3,
      Name Sprintf("Parameters/Magnet %g/3Br [T]", i)}
  ];
EndFor



For i In {1:NumSensors}
  DefineConstant[
    X~{i+NumMagnets} = {SensorPositionX~{i}*mm, Min -100*mm, Max 100*mm, Step mm, Visible 1,
      Name Sprintf("Parameters/Sensor %g/0X position [m]", i) },
    Y~{i+NumMagnets} = { SensorPositionY~{i}*mm, Min -100*mm, Max 100*mm, Step mm, Visible 1,
      Name Sprintf("Parameters/Sensor %g/0Y position [m]", i) },
    Z~{i+NumMagnets} = {0, Min -100*mm, Max 100*mm, Step mm, Visible 1,
      Name Sprintf("Parameters/Sensor %g/0Z position [m]", i) },

    M~{i+NumMagnets} = {1, Choices{1="Cube"},
      Name Sprintf("Parameters/Sensor %g/00Shape", i)},

    Lx~{i+NumMagnets} = {1*mm, Min mm, Max 100*mm, Step mm,
      Name Sprintf("Parameters/Sensor %g/1X length [m]", i),
      Visible (M~{i+NumMagnets} == 1) },
    Ly~{i+NumMagnets} = {1*mm, Min mm, Max 100*mm, Step mm, Visible Flag_FullMenu,
      Name Sprintf("Parameters/Sensor %g/1XY aspect ratio", i),
      Visible (M~{i+NumMagnets} == 1) },
    Lz~{i+NumMagnets} = {1*mm, Min mm, Max 100*mm, Step mm, Visible Flag_FullMenu,
      Name Sprintf("Parameters/Sensor %g/1XZ aspect ration", i),
      Visible (M~{i+NumMagnets} == 1) },

    Rx~{i+NumMagnets} = {0, Min -180, Max 180, Step 1,
      Name Sprintf("Parameters/Sensor %g/2X rotation [deg]", i) },
    Ry~{i+NumMagnets} = {0, Min -180, Max 180, Step 1,
      Name Sprintf("Parameters/Sensor %g/2Y rotation [deg]", i) },
    Rz~{i+NumMagnets} = {0, Min -180, Max 180, Step 1,
      Name Sprintf("Parameters/Sensor %g/2Z rotation [deg]", i) },

    MUR~{i+NumMagnets} = {1.0, Visible 1,
      Name Sprintf("Parameters/Sensor %g/3Mu relative []", i)},
    BR~{i+NumMagnets} = {0.0, Visible 1,
      Name Sprintf("Parameters/Sensor %g/3Br [T]", i)}
  ];
EndFor

//The geometrical parameters of the Infinite box.
DefineConstant[
  xInt = {1, Name "Infinite box/xInt", Visible 0}
  yInt = {1, Name "Infinite box/yInt", Visible 0}
  zInt = {1, Name "Infinite box/zInt", Visible 0}
  xExt = {xInt*2, Name "Infinite box/xExt", Visible 0}
  yExt = {yInt*2, Name "Infinite box/yExt", Visible 0}
  zExt = {zInt*2, Name "Infinite box/zExt", Visible 0}
  xCnt = {0, Name "Infinite box/xCenter", Visible 0}
  yCnt = {0, Name "Infinite box/yCenter", Visible 0}
  zCnt = {0, Name "Infinite box/zCenter", Visible 0}
];
