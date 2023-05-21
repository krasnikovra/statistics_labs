pkg load interval

input = csvread("Channel_1_500nm_0_23mm.csv")
eps = 1e-4

input_int = infsup(input - eps, input + eps)

errorbar(mid(input_int), rad(input_int), "b")
title("Data")
xlabel("n")
ylabel("mV")
xlim([0, size(input, 1)])

print -dpdfcrop ../figures/data_int.pdf

