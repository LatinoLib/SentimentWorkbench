using System;
using System.Collections.Generic;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public class MulticlassClassifier : BaseClassifier<SentimentLabel>
    {
        private SvmMulticlassClassifier<SentimentLabel> mClassifier;

        public override void Train(ILabeledExampleCollection<SentimentLabel, SparseVector<double>> dataset)
        {
            mClassifier = (SvmMulticlassClassifier<SentimentLabel>)CreateModel();
            mClassifier.Train(dataset);
        }

        public override Prediction<SentimentLabel> Predict(SparseVector<double> example)
        {
            return mClassifier.Predict(example);
        }

        protected override IEnumerable<IDisposable> GetDisposables()
        {
            return new[] { mClassifier };
        }
    }
}