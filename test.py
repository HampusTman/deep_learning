import numpy as np
test_loss = 1.8603633911378923
test_accuracy = 0.45161290322580644
correct_test = np.array([137,109,81,121])
total_test = np.array([249, 247, 248, 248])
class_names = ["Happy", "Sad", "Surprised", "Mad"]
print(f"\nFinal test results: \nloss: {np.mean(test_loss)}\n"
        f"accuracy: {np.mean(test_accuracy)}\n"
        f"The recall for ")
for i in range(len(correct_test)):
        print(f"The number of correct predictions of class {class_names[i]}: {correct_test[i]}\nTotal number of class {class_names[i]}: {total_test[i]}\n recall: {correct_test[i]/total_test[i]}\n\n")