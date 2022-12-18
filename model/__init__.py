import importlib


def find_model_using_name(model_name): 
    model_filename = "model.removal_" + model_name
    print(model_filename)
    modellib = importlib.import_module(model_filename)
    target_model_name = 'SRNet'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls
            break

    return model

def create_model(args):
    model = find_model_using_name(args.model_name)
    instance = model(args)
    print("model [%s] was created" % (instance.name()))
    return instance

