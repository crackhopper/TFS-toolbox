
class DefaultMonitor(object):
  def __init__(self,netobj):
    self.net = netobj
    self.print_step = 10

  def status(self,train,test,step,epoch):
    if step % self.print_step==0:
      acc = self.net.score(test)
      X,y = train.current_batch
      loss = self.net.measure_loss(X,y)
      print 'step %d. loss %f, test accuracy:%f'%(step,loss,acc)
