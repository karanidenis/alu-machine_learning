#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')

plt.yticks(range(0, 31, 5))
plt.xticks(range(0, 101, 10))

plt.xlim(0, 100)
plt.ylim(0, 30)
plt.hist(student_grades, bins=range(0, 101, 10),
         edgecolor='k', alpha=0.5, color='green')

plt.show()
