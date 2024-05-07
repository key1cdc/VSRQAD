
import visdom
import numpy as np

class Visualizer():
    def __init__(self):
        self.vis = visdom.Visdom(use_incoming_socket=False)
        self.win_num = 0

    def plot_loss(self, losses):    # loss = {'type' : list}    len(list) = epoch

        try:
            for (loss_type, loss) in losses:
                axis = np.linspace(1, len(loss), len(loss))
                self.vis.line(
                    Y=np.array(loss),
                    X=axis,
                    opts={
                        'title': '{} loss over epoch'.format(loss_type),
                        'xlabel': 'iteration',
                        'ylabel': '{} loss'.format(loss_type)
                    },
                    win=str(self.win_num),
                )
                self.win_num += 1

        except ConnectionError:
            print('Could not connect to Visdom server')
            exit(1)

    def display_results(self, images):     # images = {'img_name': img}
        for (label, image) in images.item():
            self.vis.images(image, win=str(self.win_num), opts=dict(title=label))
            self.win_num += 1



