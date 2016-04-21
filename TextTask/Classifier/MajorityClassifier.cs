using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public class MajorityClassifier : BaseClassifier<SentimentLabel>
    {
        //private Tuple<SentimentLabel, int>[] mLabelCounts;
        private readonly MajorityClassifier<SentimentLabel, SparseVector<double>> mInnerModel = new MajorityClassifier<SentimentLabel, SparseVector<double>>();

        public override void Train(ILabeledExampleCollection<SentimentLabel, SparseVector<double>> dataset)
        {
/*
            mLabelCounts = dataset.GroupBy(le => le.Label)
                .OrderByDescending(g => g.Count())
                .Select(g => new Tuple<SentimentLabel, int>(g.Key, g.Count()))
                .ToArray();
*/
            mInnerModel.Train(dataset);
            IsTrained = true;
        }

        public override Prediction<SentimentLabel> Predict(SparseVector<double> example)
        {
            Preconditions.CheckState(IsTrained);
            return mInnerModel.Predict(example);
/*
            Preconditions.CheckState(mLabelCounts != null && mLabelCounts.Any());
            return new Prediction<SentimentLabel>(new[] { new KeyDat<double, SentimentLabel>(1, mLabelCounts[0].Item1) });
*/
        }

        protected override IEnumerable<IDisposable> GetDisposables()
        {
            return EmptyDisposables;
        }
    }
}