using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.Model.Eval;

namespace TextTask.Classifier
{

    public class TwoPlaneClassifier : BaseClassifier<SentimentLabel>
    {
        private IModel<SentimentLabel, SparseVector<double>> mPosClassifier;
        private IModel<SentimentLabel, SparseVector<double>> mNegClassifier;

        private List<double> mPosSortedScores;
        private List<double> mNegSortedScores;
        private ExampleScore[] mExampleScores;
        private TagDistrTable<SentimentLabel> mTagDistrTable;
        private double mBinWidth;
        
        public TwoPlaneClassifier()
        {
        }

        public TwoPlaneClassifier(BinarySerializer reader)
        {
            Load(reader);
        }

        public double BiasToPosRate { get; set; }
        public double BiasToNegRate { get; set; }
        public BiasCalibration PosBiasCalibration { get; set; }
        public BiasCalibration NegBiasCalibration { get; set; }

        public bool IsScorePercentile { get; set; }

        public TagDistrTable<SentimentLabel> TagDistrTable { get { return mTagDistrTable; } }
        public double BinWidth
        {
            get { return mBinWidth; }
            set
            {
                mBinWidth = value;
                UpdateDistrTable();
            }
        }

        public override void Train(ILabeledExampleCollection<SentimentLabel, SparseVector<double>> dataset)
        {
            Preconditions.CheckNotNull(dataset);

            var posDataset = new LabeledDataset<SentimentLabel, SparseVector<double>>(dataset.Select(le => 
                new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label == SentimentLabel.Positive 
                    ? SentimentLabel.Positive : SentimentLabel.Negative, le.Example)));
            mPosClassifier = CreateModel();
            mPosClassifier.Train(posDataset);

            var negDataset = new LabeledDataset<SentimentLabel, SparseVector<double>>(dataset.Select(le => 
                new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label == SentimentLabel.Negative 
                    ? SentimentLabel.Negative : SentimentLabel.Positive, le.Example)));
            mNegClassifier = CreateModel();
            mNegClassifier.Train(negDataset);

            if (PosBiasCalibration != null || NegBiasCalibration != null)
            {
                var labeledDataset = new LabeledDataset<SentimentLabel, SparseVector<double>>(dataset);
                double? posBias = Calibrate(true, labeledDataset);
                double? negBias = Calibrate(false, labeledDataset);
                BiasToPosRate = posBias ?? BiasToPosRate;
                BiasToNegRate = negBias ?? BiasToNegRate;
            }

            mPosSortedScores = mNegSortedScores = null;
            mExampleScores = dataset.Select(le =>
            {
                Prediction<SentimentLabel> posPrediction = mPosClassifier.Predict(le.Example);
                Prediction<SentimentLabel> negPrediction = mNegClassifier.Predict(le.Example);
                return new ExampleScore
                    {
                        Label = le.Label,
                        PosScore = posPrediction.BestClassLabel == SentimentLabel.Positive ? posPrediction.BestScore : -posPrediction.BestScore,
                        NegScore = negPrediction.BestClassLabel == SentimentLabel.Negative ? -negPrediction.BestScore : negPrediction.BestScore
                    };
            }).ToArray();

            UpdateDistrTable();

            IsTrained = true;
        }

        public override Prediction<SentimentLabel> Predict(SparseVector<double> example)
        {
            Preconditions.CheckState(IsTrained);
            return PredictInternal(example, BiasToPosRate, BiasToNegRate);
        }

        public override void Save(BinarySerializer writer)
        {
            mPosClassifier.Save(writer);
            mNegClassifier.Save(writer);

            writer.WriteInt(mExampleScores.Length);
            foreach (ExampleScore es in mExampleScores)
            {
                writer.WriteInt((int)es.Label);
                writer.WriteDouble(es.PosScore);
                writer.WriteDouble(es.NegScore);
            }

            writer.WriteDouble(BinWidth);
            writer.WriteDouble(BiasToPosRate);
            writer.WriteDouble(BiasToNegRate);
            writer.WriteBool(IsScorePercentile);
            writer.WriteBool(IsTrained);
        }

        public override sealed void Load(BinarySerializer reader)
        {
            mPosClassifier = new SvmBinaryClassifier<SentimentLabel>(reader);
            mNegClassifier = new SvmBinaryClassifier<SentimentLabel>(reader);

            mExampleScores = new ExampleScore[reader.ReadInt()];
            for (int i = 0; i < mExampleScores.Length; i++)
            {
                mExampleScores[i] = new ExampleScore
                    {
                        Label = (SentimentLabel)reader.ReadInt(),
                        PosScore = reader.ReadDouble(),
                        NegScore = reader.ReadDouble()
                    };
            }

            BinWidth = reader.ReadDouble();
            BiasToPosRate = reader.ReadDouble();
            BiasToNegRate = reader.ReadDouble();
            IsScorePercentile = reader.ReadBool();
            IsTrained = reader.ReadBool();

            UpdateDistrTable();
        }

        protected override IEnumerable<IDisposable> GetDisposables()
        {
            return new[] { mPosClassifier, mNegClassifier }.Where(c => c is IDisposable).Cast<IDisposable>();
        }

        private Prediction<SentimentLabel> PredictInternal(SparseVector<double> example, double biasToPosRate, double biasToNegRate)
        {
            Prediction<SentimentLabel> posPred = mPosClassifier.Predict(example);
            Prediction<SentimentLabel> negPred = mNegClassifier.Predict(example);
            SentimentLabel negPredLabel = negPred.BestClassLabel;
            SentimentLabel posPredLabel = posPred.BestClassLabel;

            // flip over predictions with score falling under bias
            double? negScore = null, posScore = null;
            if (biasToPosRate > 0)
            {
                if (posPredLabel == SentimentLabel.Negative)
                {
                    double percentile = GetPercentileScore(posPred.BestScore, true);
                    if (percentile < biasToPosRate)
                    {
                        posPredLabel = SentimentLabel.Positive;
                        posScore = biasToPosRate - percentile;
                    }
                }
            }
            else if (biasToPosRate < 0)
            {
                if (posPredLabel == SentimentLabel.Positive)
                {
                    double score = GetPercentileScore(posPred.BestScore, true);
                    if (score < -biasToPosRate)
                    {
                        posPredLabel = SentimentLabel.Negative;
                        posScore = biasToPosRate + score;
                    }
                }
            }
            if (biasToNegRate > 0)
            {
                if (negPredLabel == SentimentLabel.Positive)
                {
                    double score = GetPercentileScore(negPred.BestScore, false);
                    if (score < biasToNegRate)
                    {
                        negPredLabel = SentimentLabel.Negative;
                        negScore = biasToNegRate - score;
                    }
                }
            }
            else if (biasToNegRate < 0)
            {
                if (negPredLabel == SentimentLabel.Negative)
                {
                    double score = GetPercentileScore(negPred.BestScore, false);
                    if (score < -biasToNegRate)
                    {
                        negPredLabel = SentimentLabel.Positive;
                        negScore = biasToNegRate + score;
                    }
                }
            }

            // determine label, calc percentile scores if needed
            SentimentLabel bestLabel = negPredLabel == posPredLabel ? negPredLabel : SentimentLabel.Neutral;
            posScore = posScore ?? (IsScorePercentile ? GetPercentileScore(posPred.BestScore, true) : posPred.BestScore);
            negScore = negScore ?? (IsScorePercentile ? GetPercentileScore(negPred.BestScore, false) : negPred.BestScore);
            
            double bestScore = Math.Min(posScore.Value, negScore.Value);
            //double bestScore = (posScore.Value + negScore.Value) / 2;

            // calc signed distance values
            posScore = posPred.BestClassLabel == SentimentLabel.Positive ? posScore : -posScore;
            negScore = negPred.BestClassLabel == SentimentLabel.Negative ? -negScore : negScore;

            if (mTagDistrTable != null)
            {
                KeyDat<double, SentimentLabel>[] labelProbs = mTagDistrTable.GetDistrValues(posScore.Value, negScore.Value)
                    .OrderByDescending(kv => kv.Value)
                    .Select(kv =>
                    {
                        double score;
                        if (kv.Value != null)
                        {
                            score = kv.Value.Value;
                        }
                        else
                        {
                            // infer from polarity
                            if (posScore > 0 && negScore > 0)
                            {
                                score = kv.Key == SentimentLabel.Positive ? 1 : 0;
                            }
                            else if (posScore < 0 && negScore < 0)
                            {
                                score = kv.Key == SentimentLabel.Negative ? 1 : 0;
                            }
                            else
                            {
                                score = kv.Key == SentimentLabel.Neutral ? 1 : 0;
                            }
                        }
                        return new KeyDat<double, SentimentLabel>(score, kv.Key);
                    }).ToArray();

                // calc entropy
                double pNeg = labelProbs.Where(kv => kv.Dat == SentimentLabel.Negative).Single().Key;
                double pPos = labelProbs.Where(kv => kv.Dat == SentimentLabel.Positive).Single().Key;
                double pNeu = labelProbs.Where(kv => kv.Dat == SentimentLabel.Neutral).Single().Key;
                //double entNeg = -pNeg * Math.Log(pNeg, 2) - (pPos + pNeu) * Math.Log(pPos + pNeu, 2);
                //double entPos = -pPos * Math.Log(pPos, 2) - (pNeg + pNeu) * Math.Log(pNeg + pNeu, 2);
                //double ent = entNeg + entPos / 2;


                Dictionary<SentimentLabel, int?> counts = mTagDistrTable.GetCounts(posScore.Value, negScore.Value);
                int count = counts.Sum(kv => kv.Value.GetValueOrDefault());
                double mean = pPos - pNeg;
                double stddev = Math.Sqrt(pNeg * (mean + 1) * (mean + 1) + pNeu * mean * mean + pPos * (mean - 1) * (mean - 1));

                double vNeg = -1 + pNeg * 2, vNeu = vNeg + pNeu * 2, maxDelta;
                if (mean <= vNeg)
                {
                    maxDelta = Math.Min(mean - -1, vNeg - mean);
                }
                else if (mean <= vNeu)
                {
                    maxDelta = Math.Min(mean - vNeg, vNeu - mean);
                }
                else
                {
                    maxDelta = Math.Min(mean - vNeu, 1 - mean);
                }
                double sem = stddev / Math.Sqrt(count);
                double zScore = maxDelta / sem;
                double semProb = double.IsNaN(zScore) ? 0 : StdErrTables.GetProb_0ToZ(zScore) * 2;


                if (count < 5 || Math.Max(posPred.BestScore, negPred.BestScore) >= 2) // go the classic way
                {
                    labelProbs = new[] { new KeyDat<double, SentimentLabel>(bestScore, bestLabel) };
                }

                return new Prediction(labelProbs)
                    {
                        PosScore = IsScorePercentile ? Math.Abs(posScore.Value) : posScore.Value,
                        NegScore = IsScorePercentile ? Math.Abs(negScore.Value) : negScore.Value,
                        HyperLabel = bestLabel,
                        ConfidenceInterval = semProb,

                        PosCount = counts.Where(kv => kv.Key == SentimentLabel.Positive).Sum(kv => kv.Value).GetValueOrDefault(),
                        NegCount = counts.Where(kv => kv.Key == SentimentLabel.Negative).Sum(kv => kv.Value).GetValueOrDefault(),
                        NeuCount = counts.Where(kv => kv.Key == SentimentLabel.Neutral).Sum(kv => kv.Value).GetValueOrDefault()
                    };
            }

            // also add info about subjectivity (to differentiate from neutral case)
            return new Prediction(new[] { new KeyDat<double, SentimentLabel>(bestScore, bestLabel) })
                {
                    PosScore = IsScorePercentile ? Math.Abs(posScore.Value) : posScore.Value,
                    NegScore = IsScorePercentile ? Math.Abs(negScore.Value) : negScore.Value
                };
        }

        private double? Calibrate(bool doPosPlane, LabeledDataset<SentimentLabel, SparseVector<double>> dataset)
        {
            BiasCalibration calibration = doPosPlane ? PosBiasCalibration : NegBiasCalibration;
            if (calibration == null) { return null; }

            Preconditions.CheckArgument(calibration.BiasStep > 0);
            Preconditions.CheckNotNull(calibration.OptimizationFunc);

            double maxScore = double.MinValue;
            double optimalBias = 0;
            var biasScorePairs = calibration.IsSaveBiasScorePairs ? new List<Tuple<double, double>>() : null;
            for (double bias = calibration.BiasLowerBound; bias <= calibration.BiasUpperBound; bias += calibration.BiasStep)
            {
                var matrix = new PerfMatrix<SentimentLabel>(null);
                foreach (LabeledExample<SentimentLabel, SparseVector<double>> le in dataset)
                {
                    Prediction<SentimentLabel> prediction = PredictInternal(le.Example, doPosPlane ? bias : 0, doPosPlane ? 0 : bias);
                    matrix.AddCount(le.Label, prediction.BestClassLabel);
                }
                double score = calibration.OptimizationFunc(matrix);
                if (score > maxScore)
                {
                    maxScore = score;
                    optimalBias = bias;
                }
                if (biasScorePairs !=  null) { biasScorePairs.Add(new Tuple<double, double>(bias, score)); }
                Console.WriteLine("{0}\t{1:0.000}\t{2:0.000}", doPosPlane, bias, score);
            }
            if (biasScorePairs != null) { calibration.BiasScorePairs = biasScorePairs.ToArray(); }

            return optimalBias;
        }

        private double GetPercentileScore(double score, bool isPosScores)
        {
            List<double> scores;
            if (isPosScores)
            {
                if (mPosSortedScores == null)
                {
                    mPosSortedScores = mExampleScores.Select(es => Math.Abs(es.PosScore)).OrderBy(s => s).ToList();
                }
                scores = mPosSortedScores;
            }
            else
            {
                if (mNegSortedScores == null)
                {
                    mNegSortedScores = mExampleScores.Select(es => Math.Abs(es.NegScore)).OrderBy(s => s).ToList();
                }
                scores = mNegSortedScores;
            }
            return (double)Math.Abs(scores.BinarySearch(score)) / scores.Count;
        }

        private void UpdateDistrTable()
        {
            if (mExampleScores == null || mBinWidth == 0)
            {
                mTagDistrTable = null;
                return;
            }

            mTagDistrTable = new EnumTagDistrTable<SentimentLabel>(2, mBinWidth, -5, 5, SentimentLabel.Exclude)
            {
                CalcDistrFunc = (tagCounts, values, tag) => ((double)tagCounts[tag] + 1) / (tagCounts.Values.Sum() + tagCounts.Count)
            };

            foreach (ExampleScore es in mExampleScores)
            {
                mTagDistrTable.AddCount(es.Label, es.PosScore, es.NegScore);
            }
            mTagDistrTable.Calculate();
        }
        
        public class BiasCalibration
        {
            public BiasCalibration()
            {
                var labels = new[] { SentimentLabel.Negative, SentimentLabel.Neutral, SentimentLabel.Positive };
                OptimizationFunc = matrix => matrix.GetF1AvgExtremeClasses(labels);
                //OptimizationFunc = matrix => matrix.GetAccuracy();
            }

            public double BiasUpperBound { get; set; }
            public double BiasLowerBound { get; set; }
            public double BiasStep { get; set; }
            public Func<PerfMatrix<SentimentLabel>, double> OptimizationFunc { get; set; }

            public double MaxScore { get; set; }
            public bool IsSaveBiasScorePairs { get; set; }
            public Tuple<double, double>[] BiasScorePairs { get; set; }
        }

        public class Prediction : Prediction<SentimentLabel>
        {
            public Prediction(IEnumerable<KeyDat<double, SentimentLabel>> classScores) : base(classScores)
            {
            }

            public double PosScore { get; set; }
            public double NegScore { get; set; }
            public double ConfidenceInterval { get; set; }
            public SentimentLabel HyperLabel { get; set; }
            public SentimentLabel ActualLabel { get; set; }

            public int PosCount { get; set; }
            public int NegCount { get; set; }
            public int NeuCount { get; set; }
        }

        public class ExampleScore
        {
            public SentimentLabel Label { get; set; }
            public double PosScore { get; set; }
            public double NegScore { get; set; }
        }
    }
}