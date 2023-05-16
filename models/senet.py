import tensorflow as tf
from tensorflow import keras

class Senet(keras.Model):

    def compile(
        self, optimizer, metrics, loss_fn,
        ):
        """ Configure the SENet model.

        Args:
            optimizer: Optimizer for the model weights
            metrics: Metrics for evaluation
            loss_fn: Loss function
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack data
        x,y_true = data
        with tf.GradientTape() as tape:
            #make prediction
            y_pred = self(x,training = True)
            loss =  self.loss_fn(y_true,y_pred)
        gradients = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
       
        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y_true, y_pred)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"loss": loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x,y_true = data

        # Compute predictions
        y_pred = self(x)

        # Update the metrics.
        self.compiled_metrics.update_state(y_true, y_pred)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        #print( self.compiled_metrics[0])
        return results
    
    def get_config(self):
        config = super(Senet,self).get_config()
        return config