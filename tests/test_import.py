#!/usr/bin/env python3

#import helpyr
from helpyr import helpyr_misc as hm
from data_wrangling import global_settings as gs

a = 'abc'
b = '123'
tf = lambda x: '' if hm.isnumeric(x) else 'not '
print("Hello world")
tf_a = tf(a)
tf_b = tf(b)
print(f"{a} is {tf_a}a number.")
print(f"{b} is {tf_b}a number.")

print(f"g = {gs.gravity}")
