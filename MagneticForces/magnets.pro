/* -------------------------------------------------------------------
   Tutorial 9 : 3D magnetostatic dual formulations and magnetic forces

   Features:
   - 3D Magnetostatics
   - Dual vector and scalar magnetic potentials formulations
   - Boundary condition at infinity with infinite elements
   - Maxwell stress tensor and rigid-body magnetic forces

   To compute the solution in a terminal:
   First generate the (3D) mesh and then run getdp with the chosen resolution
       gmsh magnets.geo -3
       getdp magnets.pro -solve MagSta_a
       OR
       getdp magnets.pro -solve MagSta_phi

   To compute the solution interactively from the Gmsh GUI:
       File > Open > magnets.pro
       Resolution can be chosen from the menu on the left:
       MagSta_a (default) or MagSta_phi
       Run (button at the bottom of the left panel)
   ------------------------------------------------------------------- */

/*
 This tutorial solves the electromagnetic field and the rigid-body forces acting
 on a set of magnetic pieces of either parallelepipedic or cylindrical shape.
 Besides position and dimension, each piece is attributed a (constant) magnetic
 permeability and/or a remanence field.  Hereafter, the pieces are all, simply
 though imprecisely, referred to as "Magnet", irresective of whether they are
 truly permanent magnets or ferromagnetic barrels.

 The tutorial model proposes two dual 3D magnetostatic formulations:

 - the magnetic vector potential formulation with spanning-tree gauging;
 - the scalar magnetic potential formulation.

 As there are no conductors, the later is rather simple. The source field "hs"
 is directly the the known coercive field hc[]:

   h = hs - grad phi   ,  hs = -hc.

 If the "Add infinite box" box is ticked, a transformation to infinity shell is
 used to impose the exact zero-field boundary condition at infinity.  See also
 Tutorial 2: magnetostatic field of an electromagnet. The shell is generated
 automatically by including "InfiniteBox.geo" at the end of the geometrical
 description of the model. It can be placed rather close of the magnets without
 loss of accuracy.

 The preferred way to compute electromagnetic forces in GetDP is as an explicit
 by-product of the Maxwell stress tensor "TM[{b}]", which is a material
 dependent function of the magnetic induction "b" field.  The magnetic force
 acting on a rigid body in empty space can be evaluated as the flux of the
 Maxwell stress tensor through a surface "S" (surrounding the body).  A special
 auxiliary function "g(S)" linked "S" is defined for each magnet, i.e.
 "g(SkinMagnet~{i}) = un~{i}".  The resultant magnetic force acting on
 "Magnet~{i}" is given by the integral:

 f~{i} = Integral [ TM[{b}] * {-grad un~{i}} ] ;

 This approach is analogous to the computation of heat flux "q(S)" through a
 surface "S" described in "Tutorial 5: thermal problem with contact
 resistances".

 Note that the Maxwell stress tensor is always discontinuous on material
 discontinuities, and that magnetic forces acting on rigid bodies depend only on
 the Maxwell stress tensor in empty space, and on the "b" and "h" field
 distribution, on the external side of "SkinMagnet~{i}" (side of the surface in
 contact with air).

 "{-grad un~{i}}" in the above formula can be regarded as the normal vector to
 "SkinMagnet~{i}" in the one element thick layer "layer~{i}" of finite elements
 around "Magnet~{i}", and "f~{i}", is thus indeed the flux of "TM[]" through the
 surface of "Magnet~{i}".

 The support of "{-grad un~{i}}" is limited to "layer~{i}", which is much
 smaller than "AirBox".  To speed up the computation of forces, a special domain
 "Vol_Force" for force integrations is defined, which contains only the layers
 "layer~{i}" of all magnets.
*/

Include "magnets_common.pro"

DefineConstant[
  // preset all getdp options and make them (in)visible
  R_ = {"MagSta_a", Name "GetDP/1ResolutionChoices", Visible 1,
	Choices {"MagSta_a", "MagSta_phi"}},
  C_ = {"-solve -v 5 -v2 -bin", Name "GetDP/9ComputeCommand", Visible 0}
  P_ = {"", Name "GetDP/2PostOperationChoices", Visible 0}
];

Group{
  // Geometrical regions (give litteral labels to geometrical region numbers)
  domInfX = Region[1];
  domInfY = Region[2];
  domInfZ = Region[3];
  AirBox  = Region[4];
  Outer   = Region[5];

  For i In {1:NumMagnets+NumSensors}
    Magnet~{i} = Region[ {(10*i)}];
    SkinMagnet~{i} = Region[ {(10*i+1)} ];
    Layer~{i} =  Region[AirBox, OnOneSideOf SkinMagnet~{i}] ;
  EndFor

  // Abstract Groups (group geometrical regions into formulation relevant groups)
  Vol_Inf = Region[ {domInfX, domInfY, domInfZ} ];
  Vol_Air = Region[ {AirBox, Vol_Inf} ];

  Vol_Magnet = Region[{}];
  Sur_Magnet = Region[{}];
  Vol_Force = Region[{}];
  For i In {1:NumMagnets+NumSensors}
    Sur_Magnet += Region[SkinMagnet~{i}];
    Vol_Magnet += Region[Magnet~{i}];
    Vol_Layer += Region[Layer~{i}];
  EndFor

  Vol_mu = Region[ {Vol_Air, Vol_Magnet}];

  Sur_Dirichlet_phi = Region[ Outer ];
  Sur_Dirichlet_a   = Region[ Outer ];

  Dom_Hgrad_phi = Region[ {Vol_Air, Vol_Magnet, Sur_Dirichlet_phi} ];
  Dom_Hcurl_a = Region[ {Vol_Air, Vol_Magnet, Sur_Dirichlet_a} ];
  Vol_Force = Region [ Vol_Layer ];
  //Vol_Force = Region [ Vol_Air ];
}

Function{
  mu0 = 4*Pi*1e-7;
  mu[ Vol_Air ] = mu0;

  For i In {1:NumMagnets+NumSensors}
    // coercive field of magnets
    DefineConstant[
      HC~{i} = {-BR~{i}/mu0,
        Name Sprintf("Parameters/Magnet %g/0Coercive magnetic field [Am^-1]", i), Visible 0}
    ];
    hc[Magnet~{i}] = Rotate[Vector[0, HC~{i}, 0], Rx~{i}, Ry~{i}, Rz~{i}];
    br[Magnet~{i}] = Rotate[Vector[0, BR~{i}, 0], Rx~{i}, Ry~{i}, Rz~{i}];
    mu[Magnet~{i}] = mu0*MUR~{i};
  EndFor

  nu[] = 1.0/mu[];

  // Maxwell stress tensor (valid for both formulations and linear materials
  TM[] = ( SquDyadicProduct[$1] - SquNorm[$1] * TensorDiag[0.5, 0.5, 0.5] ) / mu[] ;
}

Jacobian {
  { Name Vol ;
    Case {
      { Region All ; Jacobian Vol ; }
      {Region domInfX; Jacobian VolRectShell {xInt,xExt,1,xCnt,yCnt,zCnt};}
      {Region domInfY; Jacobian VolRectShell {yInt,yExt,2,xCnt,yCnt,zCnt};}
      {Region domInfZ; Jacobian VolRectShell {zInt,zExt,3,xCnt,yCnt,zCnt};}
    }
  }
}

Integration {
  { Name Int ;
    Case {
      { Type Gauss ;
        Case {
	  { GeoElement Triangle    ; NumberOfPoints 4 ; }
	  { GeoElement Quadrangle  ; NumberOfPoints 4 ; }
          { GeoElement Tetrahedron ; NumberOfPoints 4 ; }
	  { GeoElement Hexahedron  ; NumberOfPoints  6 ; }
	  { GeoElement Prism       ; NumberOfPoints  6 ; }
	}
      }
    }
  }
}

Constraint {
  { Name phi ;
    Case {
      { Region Sur_Dirichlet_phi ; Value 0. ; }
    }
  }
  { Name a ;
    Case {
      { Region Sur_Dirichlet_a ; Value 0. ; }
    }
  }
  { Name GaugeCondition_a ; Type Assign ;
    Case {
      { Region Dom_Hcurl_a ; SubRegion Sur_Dirichlet_a ; Value 0. ; }
    }
  }
  For i In {1:NumMagnets+NumSensors}
    { Name Magnet~{i} ;
      Case {
        { Region SkinMagnet~{i} ; Value 1. ; }
      }
    }
  EndFor
}

FunctionSpace {
  { Name Hgrad_phi ; Type Form0 ; // magnetic scalar potential
    BasisFunction {
      { Name sn ; NameOfCoef phin ; Function BF_Node ;
        Support Dom_Hgrad_phi ; Entity NodesOf[ All ] ; }
    }
    Constraint {
      { NameOfCoef phin ; EntityType NodesOf ; NameOfConstraint phi ; }
    }
  }
  { Name Hcurl_a; Type Form1; // magnetic vector potential
    BasisFunction {
      { Name se;  NameOfCoef ae;  Function BF_Edge;
	Support Dom_Hcurl_a ;Entity EdgesOf[ All ]; }
    }
    Constraint {
      { NameOfCoef ae;  EntityType EdgesOf ; NameOfConstraint a; }
      { NameOfCoef ae;  EntityType EdgesOfTreeIn ; EntitySubType StartingOn ;
        NameOfConstraint GaugeCondition_a ; }
    }
  }
  // auxiliary field on layer of elements touching each magnet, for the
  // accurate integration of the Maxwell stress tensor (using the gradient of
  // this field)
  For i In {1:NumMagnets+NumSensors}
    { Name Magnet~{i} ; Type Form0 ;
      BasisFunction {
        { Name sn ; NameOfCoef un ; Function BF_GroupOfNodes ;
          Support Vol_Air ; Entity GroupsOfNodesOf[ SkinMagnet~{i} ] ; }
      }
      Constraint {
        { NameOfCoef un ; EntityType GroupsOfNodesOf ; NameOfConstraint Magnet~{i} ; }
      }
    }
  EndFor
}

Formulation {
  { Name MagSta_phi ; Type FemEquation ;
    Quantity {
      { Name phi ; Type Local ; NameOfSpace Hgrad_phi ; }
      For i In {1:NumMagnets+NumSensors}
        { Name un~{i} ; Type Local ; NameOfSpace Magnet~{i} ; }
      EndFor
    }
    Equation {
      Galerkin { [ - mu[] * Dof{d phi} , {d phi} ] ;
        In Vol_mu ; Jacobian Vol ; Integration Int ; }
      Galerkin { [ - mu[] * hc[] , {d phi} ] ;
        In Vol_Magnet ; Jacobian Vol ; Integration Int ; }
      For i In {1:NumMagnets+NumSensors} // dummy term to define dofs for fully fixed space
        Galerkin { [ 0 * Dof{un~{i}} , {un~{i}} ] ;
          In Vol_Air ; Jacobian Vol ; Integration Int ; }
      EndFor
    }
  }
  { Name MagSta_a; Type FemEquation ;
    Quantity {
      { Name a  ; Type Local  ; NameOfSpace Hcurl_a ; }
      For i In {1:NumMagnets+NumSensors}
        { Name un~{i} ; Type Local ; NameOfSpace Magnet~{i} ; }
      EndFor
    }
    Equation {
      Galerkin { [ nu[] * Dof{d a} , {d a} ] ;
        In Vol_mu ; Jacobian Vol ; Integration Int ; }
      Galerkin { [ nu[] * br[] , {d a} ] ;
        In Vol_Magnet ; Jacobian Vol ; Integration Int ; }
      For i In {1:NumMagnets+NumSensors}
      // dummy term to define dofs for fully fixed space
        Galerkin { [ 0 * Dof{un~{i}} , {un~{i}} ] ;
          In Vol_Air ; Jacobian Vol ; Integration Int ; }
      EndFor
    }
  }
}

Resolution {
  { Name MagSta_phi ;
    System {
      { Name A ; NameOfFormulation MagSta_phi ; }
    }
    Operation {
      Generate[A] ; Solve[A] ; SaveSolution[A] ;
      PostOperation[MagSta_phi] ;
    }
  }
  { Name MagSta_a ;
    System {
      { Name A ; NameOfFormulation MagSta_a ; }
    }
    Operation {
      Generate[A] ; Solve[A] ; SaveSolution[A] ;
      PostOperation[MagSta_a] ;
    }
  }
}

PostProcessing {
  { Name MagSta_phi ; NameOfFormulation MagSta_phi ;
    Quantity {
      { Name b   ;
	Value { Local { [ - mu[] * {d phi} ] ; In Dom_Hgrad_phi ; Jacobian Vol ; }
	        Local { [ - mu[] * hc[] ]    ; In Vol_Magnet ; Jacobian Vol ; } } }
      { Name h   ;
	Value { Local { [ - {d phi} ]        ; In Dom_Hgrad_phi ; Jacobian Vol ; } } }
      { Name hc  ; Value { Local { [ hc[] ]  ; In Vol_Magnet ; Jacobian Vol ; } } }
      { Name phi ; Value { Local { [ {phi} ] ; In Dom_Hgrad_phi ; Jacobian Vol ; } } }
      For i In {1:NumMagnets+NumSensors}
        { Name un~{i} ; Value { Local { [ {un~{i}} ] ; In Vol_Force ; Jacobian Vol ; } } }
        { Name f~{i} ; Value { Integral { [ - TM[-mu[] * {d phi}] * {d un~{i}} ] ;
              In Vol_Force ; Jacobian Vol ; Integration Int ; } } }
        { Name fx~{i} ; Value { Integral { [ CompX[- TM[-mu[] * {d phi}] * {d un~{i}} ] ] ;
              In Vol_Force ; Jacobian Vol ; Integration Int ; } } }
        { Name fy~{i} ; Value { Integral { [ CompY[- TM[-mu[] * {d phi}] * {d un~{i}} ] ] ;
              In Vol_Force ; Jacobian Vol ; Integration Int ; } } }
        { Name fz~{i} ; Value { Integral { [ CompZ[- TM[-mu[] * {d phi}] * {d un~{i}} ] ] ;
              In Vol_Force ; Jacobian Vol ; Integration Int ; } } }
      EndFor
    }
  }
  { Name MagSta_a ; NameOfFormulation MagSta_a ;
    PostQuantity {
      { Name b ; Value { Local { [ {d a} ]; In Dom_Hcurl_a ; Jacobian Vol; } } }
      { Name a ; Value { Local { [ {a} ]; In Dom_Hcurl_a ; Jacobian Vol; } } }
      { Name br ; Value { Local { [ br[] ]; In Vol_Magnet ; Jacobian Vol; } } }
      For i In {1:NumMagnets+NumSensors}
        { Name un~{i} ; Value { Local { [ {un~{i}} ] ; In Dom_Hcurl_a ; Jacobian Vol ; } } }
        { Name f~{i} ; Value { Integral { [ - TM[{d a}] * {d un~{i}} ] ;
              In Vol_Force ; Jacobian Vol ; Integration Int ; } } }
        { Name fx~{i} ; Value { Integral { [ CompX[- TM[{d a}] * {d un~{i}} ] ] ;
              In Vol_Force ; Jacobian Vol ; Integration Int ; } } }
        { Name fy~{i} ; Value { Integral { [ CompY[- TM[{d a}] * {d un~{i}} ] ] ;
              In Vol_Force ; Jacobian Vol ; Integration Int ; } } }
        { Name fz~{i} ; Value { Integral { [ CompZ[- TM[{d a}] * {d un~{i}} ] ] ;
              In Vol_Force ; Jacobian Vol ; Integration Int ; } } }
      EndFor
    }
  }
}

PostOperation {
  { Name MagSta_phi ; NameOfPostProcessing MagSta_phi;
    Operation {
      Print[ b, OnElementsOf Vol_mu, File "b.pos" ] ;
      Echo[ Str["l=PostProcessing.NbViews-1;",
		"View[l].ArrowSizeMax = 100;",
		"View[l].CenterGlyphs = 1;",
		"View[l].VectorType = 1;" ] ,
        File "tmp.geo", LastTimeStepOnly] ;
      /*For i In {1:NumMagnets+NumSensors}
        Print[ f~{i}[Vol_Air], OnGlobal, Format Table, File > "F.dat"  ];
        Print[ fx~{i}[Vol_Air], OnGlobal, Format Table, File > "Fx.dat",
          SendToServer Sprintf("Output/Magnet %g/X force [N]", i), Color "Ivory"  ];
        Print[ fy~{i}[Vol_Air], OnGlobal, Format Table, File > "Fy.dat",
          SendToServer Sprintf("Output/Magnet %g/Y force [N]", i), Color "Ivory"  ];
        Print[ fz~{i}[Vol_Air], OnGlobal, Format Table, File > "Fz.dat",
          SendToServer Sprintf("Output/Magnet %g/Z force [N]", i), Color "Ivory"  ];
      EndFor*/
    }
  }
  { Name MagSta_a ; NameOfPostProcessing MagSta_a ;
    Operation {
      Print[ b,  OnElementsOf Vol_mu,  File "b.pos" ];
      Echo[ Str["l=PostProcessing.NbViews-1;",
		"View[l].ArrowSizeMax = 100;",
		"View[l].CenterGlyphs = 1;",
		"View[l].VectorType = 1;" ] ,
	    File "tmp.geo", LastTimeStepOnly] ;
      //Print[ br,  OnElementsOf Vol_Magnet,  File "br.pos" ];
      //Print[ a,  OnElementsOf Vol_mu,  File "a.pos" ];
      /*For i In {1:NumMagnets+NumSensors}
      //Print[ un~{i}, OnElementsOf Domain, File "un.pos"  ];
        Print[ f~{i}[Vol_Air], OnGlobal, Format Table, File > "F.dat"  ];
        Print[ fx~{i}[Vol_Air], OnGlobal, Format Table, File > "Fx.dat",
          SendToServer Sprintf("Output/Magnet %g/X force [N]", i), Color "Ivory"  ];
        Print[ fy~{i}[Vol_Air], OnGlobal, Format Table, File > "Fy.dat",
          SendToServer Sprintf("Output/Magnet %g/Y force [N]", i), Color "Ivory"  ];
        Print[ fz~{i}[Vol_Air], OnGlobal, Format Table, File > "Fz.dat",
          SendToServer Sprintf("Output/Magnet %g/Z force [N]", i), Color "Ivory"  ];
      EndFor*/
    }
  }
}
