from .ppt import PPT_model
def create_model(opt):
    model = None
    model = PPT_model()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
