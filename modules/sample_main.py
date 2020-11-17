import module

container = module.Container("../configs/baseline_detection.aarconfig")

container.load_configuration()

exit = False

while(exit == False):
    exit = container.tick()

container.close()