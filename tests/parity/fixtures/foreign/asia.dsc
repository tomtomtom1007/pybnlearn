belief network "unknown"
node A {
  type : discrete [ 2 ] = { "no", "yes" };
}
node B {
  type : discrete [ 2 ] = { "no", "yes" };
}
node D {
  type : discrete [ 2 ] = { "no", "yes" };
}
node E {
  type : discrete [ 2 ] = { "no", "yes" };
}
node L {
  type : discrete [ 2 ] = { "no", "yes" };
}
node S {
  type : discrete [ 2 ] = { "no", "yes" };
}
node T {
  type : discrete [ 2 ] = { "no", "yes" };
}
node X {
  type : discrete [ 2 ] = { "no", "yes" };
}
probability ( A ) {
   0.9916, 0.0084;
}
probability ( B | S ) {
  (0) : 0.7006036, 0.2993964;
  (1) : 0.2823062, 0.7176938;
}
probability ( D | B, E ) {
  (0, 0) : 0.90017286, 0.09982714;
  (1, 0) : 0.2137306, 0.7862694;
  (0, 1) : 0.2773723, 0.7226277;
  (1, 1) : 0.1459227, 0.8540773;
}
probability ( E | L, T ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.0, 1.0;
  (1, 1) : 0.0, 1.0;
}
probability ( L | S ) {
  (0) : 0.98631791, 0.01368209;
  (1) : 0.8823062, 0.1176938;
}
probability ( S ) {
   0.497, 0.503;
}
probability ( T | A ) {
  (0) : 0.991528842, 0.008471158;
  (1) : 0.95238095, 0.04761905;
}
probability ( X | E ) {
  (0) : 0.95658747, 0.04341253;
  (1) : 0.005405405, 0.994594595;
}
