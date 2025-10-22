(* Load entropy data (no header) *)
entropyData = Import[
  FileNameJoin[{"..", "data", "delta=1.5_renyi_grid.csv"}], "CSV"];

(* Extract entropy values (column 3) *)
entropyValues = entropyData[[All, 3]];

(* Import spin data *)
spinData = Import[
  FileNameJoin[{"..", "data", "delta=1.5_spin_texture.csv"}], "CSV"];

(* Drop header row if present *)
If[! NumericQ[spinData[[1, 1]]], spinData = Rest@spinData];

(* Split into lattice points and spin vectors *)
spacingFactor = 1.0;
pts2 = spacingFactor*spinData[[All, {1, 2}]];
vecs = spinData[[All, {3, 4, 5}]];
pts3 = Append[#, 0] & /@ pts2;  (* Lift into 3D *)

(* Arrow parameters *)
scale = 1.8;
shaftFrac = 0.5;
headFrac = 1 - shaftFrac;
shaftRadius = 0.06;
headRadius = 0.15;

minE = Min[entropyValues];
maxE = Max[entropyValues];

(* Map vectors to arrows with entropy coloring *)
arrows = MapThread[
   Module[{pt, v, dir, shaftEnd, color},
     v = scale #2; pt = #1; dir = v;
     shaftEnd = pt + shaftFrac dir;
     color = ColorData["TemperatureMap"][
       Rescale[#3, {minE, maxE}, {0, 1}]
       ];
     {EdgeForm[], FaceForm[color],
      Cylinder[{pt, shaftEnd}, shaftRadius],
      Cone[{shaftEnd, pt + dir}, headRadius]}
   ] &,
   {pts3, vecs, entropyValues}
   ];

(* Legend *)
legend = BarLegend[
   {"TemperatureMap", {minE, maxE}},
   LegendLabel -> Placed[
     Style["Rényi-2 Entropy", 12], Left, Rotate[#, 90 Degree] &],
   LabelStyle -> {FontSize -> 12, Black},
   TicksStyle -> Black
   ];

(* Display graphics and legend *)
GraphicsRow[
  {Graphics3D[arrows, PlotRange -> All, Lighting -> "None",
    Boxed -> False, ImageSize -> 500], legend}
]
