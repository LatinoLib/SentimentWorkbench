using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.Model.Eval;

namespace TextTask.Classifier
{
    public class NeutralZoneClassifier : BaseClassifier<SentimentLabel>
    {
        private IModel<SentimentLabel> mBinaryClassifier;
        private double? mNegCentile;
        private double? mPosCentile;

        public NeutralZoneClassifier()
        {
            NumTrainFolds = 2;

            IsCalcStats = false;
            IsCalcBounds = false;
        }

        public int NumTrainFolds { get; set; }

        public double? Centile { get; set; }
        public double? NegCentile
        {
            get { return mNegCentile ?? Centile; }
            set { mNegCentile = value; }
        }
        public double? PosCentile
        {
            get { return mPosCentile ?? Centile; }
            set { mPosCentile = value; }
        }

        public bool IsCalcBounds { get; set; }
        public bool IsCalcStats { get; set; }
        public Stats TrainStats { get; private set; }

        public double PosBound { get; private set; }
        public double NegBound { get; private set; }

        public override void Train(ILabeledExampleCollection<SentimentLabel, SparseVector<double>> dataset)
        {
            Preconditions.CheckNotNull(dataset);
            Preconditions.CheckArgumentRange(IsCalcBounds || NegCentile >= 0 && NegCentile <= 1);
            Preconditions.CheckArgumentRange(IsCalcBounds || PosCentile >= 0 && PosCentile <= 1);

            var labeledDataset = (LabeledDataset<SentimentLabel, SparseVector<double>>)dataset;

            if (labeledDataset.Count == 0)
            {
                Console.WriteLine("empty dataset");
            }

            TrainStats = null;

            var posScores = new List<double>();
            var negScores = new List<double>();
            var neutralScores = new List<double>();
            var trainDataset = new LabeledDataset<SentimentLabel, SparseVector<double>>(labeledDataset.Where(le => le.Label != SentimentLabel.Neutral));
            var neutralDataset = IsCalcStats || IsCalcBounds 
                ? new LabeledDataset<SentimentLabel, SparseVector<double>>(dataset.Where(le => le.Label == SentimentLabel.Neutral)) 
                : null;

            var validation = new CrossValidator<SentimentLabel, SparseVector<double>>
            {
                NumFolds = NumTrainFolds,
                Dataset = trainDataset,

                OnAfterPrediction = (sender, foldN, model, example, le, prediction) =>
                {
                    if (le.Label == prediction.BestClassLabel)
                    {
                        if (le.Label == SentimentLabel.Positive)
                        {
                            posScores.Add(prediction.BestScore);
                        }
                        else
                        {
                            negScores.Add(-prediction.BestScore);
                        }
                    }
                    return true;
                },

                OnAfterFold = (sender, foldN, trainSet, testSet) =>
                {
                    if (IsCalcStats || IsCalcBounds)
                    {
                        neutralScores.AddRange(neutralDataset
                            .Select(le => sender.Models[0].Predict(le.Example))
                            .Select(p => p.BestClassLabel == SentimentLabel.Positive ? p.BestScore : -p.BestScore));
                    }
                }
            };
            validation.Models.Add(CreateModel());
            validation.Run();

            if (IsCalcBounds)
            {
                double negMaxProb, negScore;
                NegBound = FindMaxExclusiveProbability(neutralScores.Where(s => s < 0).Select(s => -s), 
                    negScores.Select(s => -s), out negMaxProb, out negScore) ? -negScore : 0;

                double posMaxProb, posScore;
                PosBound = FindMaxExclusiveProbability(neutralScores.Where(s => s > 0), 
                    posScores, out posMaxProb, out posScore) ? posScore : 0;
            }
            else
            {
                if (NegCentile != null)
                {
                    NegBound = negScores.OrderByDescending(bs => bs).Skip((int)Math.Truncate(negScores.Count * NegCentile.Value)).FirstOrDefault();
                }
                if (PosCentile != null)
                {
                    PosBound = posScores.OrderBy(bs => bs).Skip((int)Math.Truncate(posScores.Count * PosCentile.Value)).FirstOrDefault();
                }
            }

            if (IsCalcStats) { TrainStats = CalcStats(negScores, neutralScores, posScores); }

            mBinaryClassifier = validation.Models[0];
            mBinaryClassifier.Train(trainDataset);

            IsTrained = true;
        }


        public override Prediction<SentimentLabel> Predict(SparseVector<double> example)
        {
            Preconditions.CheckState(IsTrained);

            Prediction<SentimentLabel> prediction = mBinaryClassifier.Predict(example);
            SentimentLabel bestLabel = prediction.BestClassLabel;
            double bestScore = bestLabel == SentimentLabel.Negative ? -prediction.BestScore : prediction.BestScore;
            if (bestScore > NegBound && bestScore < PosBound)
            {
                bestLabel = SentimentLabel.Neutral;
            }
            return new Prediction<SentimentLabel>(new[] { new KeyDat<double, SentimentLabel>(bestScore, bestLabel) });
        }

        protected override IEnumerable<IDisposable> GetDisposables()
        {
            return new[] { mBinaryClassifier }.Where(c => c is IDisposable).Cast<IDisposable>();
        }

        private static bool FindMaxExclusiveProbability(IEnumerable<double> scores, IEnumerable<double> notScores, out double maxProb, out double maxDifScore)
        {
            double[] s1 = Preconditions.CheckNotNull(scores).OrderBy(s => s).ToArray();
            double[] s2 = Preconditions.CheckNotNull(notScores).OrderBy(s => s).ToArray();

            int count = s1.Count();
            int notCount = s2.Count();
            if (count == 0 || notCount == 0)
            {
                maxProb = double.NaN; 
                maxDifScore = double.NaN;
                return false;
            }

            maxProb = 0; maxDifScore = 0;
            for (int i = 0, j = 0; i < count; i++)
            {
                while (j < notCount && s2[j] < s1[i]) { j++; }
                if (j == notCount) { break; }
                double p = (double)(i + 1) / count;
                double notP = 1 - (double)(j + 1) / notCount;

                // of two probabilities for the similar score: p1 & !p2
                double dif = p * notP;
                if (dif > maxProb)
                {
                    maxProb = dif;
                    maxDifScore = s1[i];
                }
            }

            return true;
        }

        private Stats CalcStats(List<double> negScores, List<double> neutralScores, List<double> posScores)
        {
            return new Stats
                {
                    NegScores = negScores.ToArray(),
                    PosScores = posScores.ToArray(),
                    NeutralScores = neutralScores.ToArray(),

                    PosAsNuetralErr = (double)posScores.Count(s => s < PosBound) / posScores.Count,
                    NegAsNuetralErr = (double)negScores.Count(s => s > NegBound) / negScores.Count,
                    NuetralAsPosErr = (double)neutralScores.Count(s => s >= PosBound) / neutralScores.Count,
                    NuetralAsNegErr = (double)neutralScores.Count(s => s <= NegBound) / neutralScores.Count
                };
        }

        public class Stats
        {
            public double[] NegScores { get; set; }
            public double[] PosScores { get; set; }
            public double[] NeutralScores { get; set; }

            public double PosAsNuetralErr { get; set; }
            public double NegAsNuetralErr { get; set; }
            public double NuetralAsNegErr { get; set; }
            public double NuetralAsPosErr { get; set; }
        }
    }
}