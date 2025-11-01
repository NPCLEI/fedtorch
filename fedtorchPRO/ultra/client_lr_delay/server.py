from fedtorchPRO import FedADAM

class CLD(FedADAM):

    def aggregate(self):
        if self.cur_comic > 2000 and self.cur_comic % 100 == 0:
            self.arguments['client_trainer_arguments']['lr'] = self.arguments['client_trainer_arguments']['lr'] / 5

        return super().aggregate()