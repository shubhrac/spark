package org.apache.spark.mllib.tree.loss

/**
  * Created by shuchandra on 2/17/16.
  */

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.annotation.Since
import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.TreeEnsembleModel


/**
  * :: DeveloperApi ::
  * Class for Quantile Loss calculation (for regression).
  *
  * The Quantile loss function is defined as:
  *  t.(F(x)-y)  when F(x) >= y
  *  (1-t) (y - F(x)) when F(x) < y
  * where y is the label and F(x) is the model prediction for features x.
  */

@DeveloperApi
object QuantileLoss extends Loss {

  //TO DO: Needs to be passed as a parameter
  var t = 0.6

  /**
    * Method to calculate the gradients for the gradient boosting calculation for Quantile loss error calculation
    * The gradient with respect to F(x) is:
    *    t   when F(x) >= y
    *    t-1  when F(x) < y
    * @param prediction Predicted label.
    * @param label True label.
    * @return Loss gradient
    */
  @Since("1.2.0")
  override def gradient(prediction: Double, label: Double): Double = {
    if (label - prediction > 0) (t-1) else t
  }

  override private[mllib] def computeError(prediction: Double, label: Double): Double = {
    if(label - prediction > 0) (1-t)*(label - prediction) else  t*(prediction - label)
  }

  def getQuantileParameter :Double = {
    this.t
  }
  
  def setQuantileParameter(quantile: Double) = {
    this.t= quantile
  }
}
