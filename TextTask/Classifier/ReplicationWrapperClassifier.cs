using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.TextMining;

namespace TextTask.Classifier
{
    public class ReplicationWrapperClassifier : BaseClassifier<SentimentLabel>
    {
        private IModel<SentimentLabel> mClassifier;

        public ReplicationWrapperClassifier()
        {
            H1 = 0;
            H2 = 1;
        }

        public double H1 { get; set; }
        public double H2 { get; set; }
        public BowSpace BowSpace { get; set; }

        public override void Train(ILabeledExampleCollection<SentimentLabel, SparseVector<double>> dataset)
        {
            Preconditions.CheckState(BowSpace != null);
            var replDataset = new LabeledDataset<SentimentLabel, SparseVector<double>>();
            foreach (LabeledExample<SentimentLabel, SparseVector<double>> le in dataset)
            {
                SparseVector<double> vector1, vector2;
                Replicate(le.Example, out vector1, out vector2);

                replDataset.Add(new LabeledExample<SentimentLabel, SparseVector<double>>(
                    le.Label == SentimentLabel.Neutral ? SentimentLabel.Negative : le.Label, vector1));
                replDataset.Add(new LabeledExample<SentimentLabel, SparseVector<double>>(
                    le.Label == SentimentLabel.Neutral ? SentimentLabel.Positive : le.Label, vector2));
            }

            mClassifier = CreateModel();
            mClassifier.Train(replDataset);

            IsTrained = true;
        }

        public override Prediction<SentimentLabel> Predict(SparseVector<double> example)
        {
            Preconditions.CheckState(BowSpace != null);
            Preconditions.CheckState(IsTrained);

            SparseVector<double> vector1, vector2;
            Replicate(example, out vector1, out vector2);

            Prediction<SentimentLabel> pred1 = mClassifier.Predict(vector1);
            Prediction<SentimentLabel> pred2 = mClassifier.Predict(vector2);

            var bestLabel = SentimentLabel.Neutral;
            if (pred1.BestClassLabel == SentimentLabel.Positive && pred2.BestClassLabel == SentimentLabel.Positive)
            {
                bestLabel = SentimentLabel.Positive;
            }
            else if (pred1.BestClassLabel == SentimentLabel.Negative && pred2.BestClassLabel == SentimentLabel.Negative)
            {
                bestLabel = SentimentLabel.Negative;
            }
            double bestScore = Math.Abs(pred1.BestScore) + Math.Abs(pred2.BestScore) / 2;

            return new Prediction<SentimentLabel>(new[] { new KeyDat<double, SentimentLabel>(bestScore, bestLabel) });
        }

        protected override IEnumerable<IDisposable> GetDisposables()
        {
            return new[] { mClassifier }.Where(c => c is IDisposable).Cast<IDisposable>();
        }

        private void Replicate(SparseVector<double> vector, out SparseVector<double> vector1, out SparseVector<double> vector2)
        {
            int featureIdx = BowSpace.Words.Count;
            vector1 = vector.Clone();
            vector1.InnerIdx.Add(featureIdx);
            vector1.InnerDat.Add(H1);
            
            vector2 = vector.Clone();
            vector2.InnerIdx.Add(featureIdx);
            vector2.InnerDat.Add(H2);

            ModelUtils.TryNrmVecL2(vector1);
            ModelUtils.TryNrmVecL2(vector2);
        }
    }
}