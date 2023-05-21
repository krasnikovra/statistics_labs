xlimits = [1 size(input, 1)]

hold on
ir_plotmodelset(irproblem, [-50 250])
errorbar(input, max(w2) * eps, "b")

xlim(xlimits)
ylim([input(1) - max(w2) * eps, input(end) + max(w2) * eps])

title("Corridor of joint dependencies")
xlabel("n")
ylabel("mV")

print -dpdfcrop ../figures/corridor_of_joint_dep.pdf

xlim("auto")
ylim("auto")

print -dpdfcrop ../figures/corridor_of_joint_dep_zoomout.pdf

