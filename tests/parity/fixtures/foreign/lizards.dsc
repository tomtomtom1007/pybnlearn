belief network "unknown"
node Diameter {
  type : discrete [ 2 ] = { "narrow", "wide" };
}
node Height {
  type : discrete [ 2 ] = { "high", "low" };
}
node Species {
  type : discrete [ 2 ] = { "Sagrei", "Distichus" };
}
probability ( Diameter | Species ) {
  (0) : 0.7195122, 0.2804878;
  (1) : 0.5469388, 0.4530612;
}
probability ( Height | Species ) {
  (0) : 0.7378049, 0.2621951;
  (1) : 0.5836735, 0.4163265;
}
probability ( Species ) {
   0.400978, 0.599022;
}
