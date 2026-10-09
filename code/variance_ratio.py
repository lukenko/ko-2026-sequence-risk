# Five-year variance ratio of real equity returns, and the AR(1) phi that matches it.
import numpy as np
from historical import real_series

def VR(x, k):
    agg = np.array([x[i:i + k].sum() for i in range(len(x) - k + 1)])
    return agg.var() / (k * x.var())

def VR_ar1(phi, k=5):
    return 1 + 2 * sum((1 - j / k) * phi ** j for j in range(1, k))

eq = real_series(1.00)   # all-equity real returns, 1928-2025
print("Variance ratio, U.S. real equity returns 1928-2025:")
print(f"  VR(5) = {VR(eq, 5):.2f}   (VR(3) = {VR(eq, 3):.2f})")
print("AR(1) variance ratio VR(5) by phi:")
for phi in (-0.15, -0.20, -0.25, -0.30):
    print(f"  phi = {phi:+.2f}  ->  VR(5) = {VR_ar1(phi):.2f}")
