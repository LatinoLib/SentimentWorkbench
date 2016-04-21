using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public class NeutralZoneReliabilityClassifier : BaseClassifier<SentimentLabel>
    {
        private IModel<SentimentLabel> mBinaryClassifier;
        private double mReliabilityThreshold = 0.5;

        public NeutralZoneReliabilityClassifier()
        {
        }

        public NeutralZoneReliabilityClassifier(IModel<SentimentLabel> binaryClassifier, ILabeledExampleCollection<SentimentLabel, SparseVector<double>> dataset)
        {
            mBinaryClassifier = binaryClassifier;
            Train(dataset);
        }

        public NeutralZoneReliabilityClassifier(BinarySerializer reader)
        {
            Load(reader);
        }

        public double NegAverageDistance { get; private set; }
        public double PosAverageDistance { get; private set; }

        public double ReliabilityThreshold
        {
            get { return mReliabilityThreshold; }
            set { mReliabilityThreshold = value; }
        }

        public override sealed void Train(ILabeledExampleCollection<SentimentLabel, SparseVector<double>> dataset)
        {
            Preconditions.CheckNotNull(dataset);

            var labeledDataset = (LabeledDataset<SentimentLabel, SparseVector<double>>)dataset;

            var trainDataset = new LabeledDataset<SentimentLabel, SparseVector<double>>(labeledDataset.Where(le => le.Label != SentimentLabel.Neutral));

            if (mBinaryClassifier == null)
            {
                mBinaryClassifier = CreateModel();
                mBinaryClassifier.Train(trainDataset);
            }

            IsTrained = true;

            /*Calculate positive and negative average distances*/
            int positiveTweetsNumber = 0;
            int negativeTweetsNumber = 0;
            PosAverageDistance = 0;
            NegAverageDistance = 0;

            foreach (LabeledExample<SentimentLabel, SparseVector<double>> example in trainDataset)
            {
                Prediction<SentimentLabel> prediction = mBinaryClassifier.Predict(example.Example);
                SentimentLabel bestLabelPredicted = prediction.BestClassLabel;
                double bestScorePredicted = bestLabelPredicted == SentimentLabel.Negative ? -prediction.BestScore : prediction.BestScore;

                SentimentLabel actualLabel = example.Label;

                if (actualLabel == SentimentLabel.Positive)
                {
                    PosAverageDistance += bestScorePredicted;
                    positiveTweetsNumber++;
                }
                else if (actualLabel == SentimentLabel.Negative)
                {
                    NegAverageDistance += bestScorePredicted;
                    negativeTweetsNumber++;
                }
            }

            PosAverageDistance = PosAverageDistance / positiveTweetsNumber;
            NegAverageDistance = NegAverageDistance / negativeTweetsNumber;
        }

        public override Prediction<SentimentLabel> Predict(SparseVector<double> example)
        {
            Preconditions.CheckState(IsTrained);

            Prediction<SentimentLabel> prediction = mBinaryClassifier.Predict(example);
            SentimentLabel bestLabel = prediction.BestClassLabel;
            double bestScore = bestLabel == SentimentLabel.Negative ? -prediction.BestScore : prediction.BestScore;

            double reliability = bestScore / (2.0 * ((bestScore) > 0.0 ? PosAverageDistance : NegAverageDistance));
            if (reliability > 1.0) { reliability = 1.0; }

            SentimentLabel predictedLabel;

            if (bestScore > 0 && reliability >= ReliabilityThreshold)
            {
                predictedLabel = SentimentLabel.Positive;
            }            
            else if (bestScore <= 0 && reliability >= ReliabilityThreshold)
            {
                predictedLabel = SentimentLabel.Negative;
            }
            else
            {
                predictedLabel = SentimentLabel.Neutral;
            }

            return new Prediction<SentimentLabel>(new[] { new KeyDat<double, SentimentLabel>(bestScore, predictedLabel) });
        }

        public override void Save(BinarySerializer writer)
        {
            mBinaryClassifier.Save(writer);
            writer.WriteDouble(NegAverageDistance);
            writer.WriteDouble(PosAverageDistance);
            writer.WriteDouble(mReliabilityThreshold);
            writer.WriteBool(IsTrained);
        }

        public override sealed void Load(BinarySerializer reader)
        {
            mBinaryClassifier = new SvmBinaryClassifier<SentimentLabel>(reader);
            NegAverageDistance = reader.ReadDouble();
            PosAverageDistance = reader.ReadDouble();
            mReliabilityThreshold = reader.ReadDouble();
            IsTrained = reader.ReadBool();
        }

        protected override IEnumerable<IDisposable> GetDisposables()
        {
            return new[] { mBinaryClassifier }.Where(c => c is IDisposable).Cast<IDisposable>();
        }
    }
}