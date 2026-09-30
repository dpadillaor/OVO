// Comprueba que el build y GoogleTest funcionan. Bórralo cuando exista el primer test real.
#include <gtest/gtest.h>

TEST(Build, GoogleTestWorks) { EXPECT_EQ(1 + 1, 2); }
