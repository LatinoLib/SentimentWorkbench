using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public class ThreePlaneOneVsOneBinVotingClassifier : BinVotingClassifier<SentimentLabel, SparseVector<double>>
    {
        public ThreePlaneOneVsOneBinVotingClassifier(double binWidth = 0.05)
            : base(new IModel<SentimentLabel, SparseVector<double>>[3], 0.5, SentimentLabel.Exclude)
        {
        }

        public ThreePlaneOneVsOneBinVotingClassifier(IModel<SentimentLabel, SparseVector<double>>[] innerModels, double binWidth = 0.05) 
            : base(innerModels, binWidth)
        {
            Preconditions.CheckNotNull(innerModels);
            Preconditions.CheckArgument(innerModels.Length == 3);
        }

        protected override IEnumerable<double> GetPredictionScores(Prediction<SentimentLabel>[] predictions)
        {
            var scores = new double[3];
            scores[0] = predictions[0].BestClassLabel == SentimentLabel.Negative ? -predictions[0].BestScore : predictions[0].BestScore;
            scores[1] = predictions[1].BestClassLabel == SentimentLabel.Neutral ? -predictions[1].BestScore : predictions[1].BestScore;
            scores[2] = predictions[2].BestClassLabel == SentimentLabel.Negative ? -predictions[2].BestScore : predictions[2].BestScore;
            return scores;
        }

        protected override LabeledDataset<SentimentLabel, SparseVector<double>> GetTrainSet(int modelIdx, 
            IModel<SentimentLabel, SparseVector<double>> model, LabeledDataset<SentimentLabel, SparseVector<double>> trainSet)
        {
            switch (modelIdx)
            {
                case 0: return new LabeledDataset<SentimentLabel, SparseVector<double>>(trainSet.Where(le => le.Label != SentimentLabel.Neutral));
                case 1: return new LabeledDataset<SentimentLabel, SparseVector<double>>(trainSet.Where(le => le.Label != SentimentLabel.Negative));
                case 2: return new LabeledDataset<SentimentLabel, SparseVector<double>>(trainSet.Where(le => le.Label != SentimentLabel.Positive));
                default: throw new Exception();
            }
        }

        protected override double CalcLabelScore(Dictionary<SentimentLabel, int> labelCounts, double[] modelScores, SentimentLabel label)
        {
            double sum = labelCounts.Values.Sum();
            double pNeg = labelCounts.Single(kv => kv.Key == SentimentLabel.Negative).Value / sum;
            double pPos = labelCounts.Single(kv => kv.Key == SentimentLabel.Positive).Value / sum;
            double pNeu = labelCounts.Single(kv => kv.Key == SentimentLabel.Neutral).Value / sum;

            double entNeg = -pNeg * Math.Log(pNeg, 2) - (pPos + pNeu) * Math.Log(pPos + pNeu, 2);
            double entPos = -pPos * Math.Log(pPos, 2) - (pNeg + pNeu) * Math.Log(pNeg + pNeu, 2);
            double ent = entNeg + entPos / 2;

            if (ent > 1)
            {
                return label == SentimentLabel.Neutral ? 1 : 0;
            }
            switch (label)
            {
                case SentimentLabel.Negative : return pNeg;
                case SentimentLabel.Neutral : return pNeu;
                case SentimentLabel.Positive : return pPos;
                default:
                    throw new Exception();
            }
        }
    }
}