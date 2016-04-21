using System;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public class ThreePlaneOneVsAllVotingClassifier : VotingClassifier<SentimentLabel, SparseVector<double>>
    {
        public ThreePlaneOneVsAllVotingClassifier()
            : base(new IModel<SentimentLabel, SparseVector<double>>[3])
        {
        }

        public ThreePlaneOneVsAllVotingClassifier(IModel<SentimentLabel, SparseVector<double>>[] innerModels)
            : base(innerModels)
        {
            Preconditions.CheckNotNull(innerModels);
            Preconditions.CheckArgument(innerModels.Length == 3);
        }

        public ThreePlaneOneVsAllVotingClassifier(BinarySerializer reader)
            : base(reader)
        {
        }

        protected override LabeledDataset<SentimentLabel, SparseVector<double>> GetTrainSet(int modelIdx, IModel<SentimentLabel,
            SparseVector<double>> model, LabeledDataset<SentimentLabel, SparseVector<double>> trainSet)
        {
            switch (modelIdx)
            {
                case 0: return new LabeledDataset<SentimentLabel, SparseVector<double>>(trainSet.Select(le =>
                    new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label == SentimentLabel.Neutral ? SentimentLabel.Neutral : SentimentLabel.Negative, le.Example)));
                case 1: return new LabeledDataset<SentimentLabel, SparseVector<double>>(trainSet.Select(le =>
                    new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label == SentimentLabel.Negative ? SentimentLabel.Negative : SentimentLabel.Positive, le.Example)));
                case 2: return new LabeledDataset<SentimentLabel, SparseVector<double>>(trainSet.Select(le =>
                    new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label == SentimentLabel.Positive ? SentimentLabel.Positive : SentimentLabel.Neutral, le.Example)));
                default: throw new Exception();
            }
        }

        protected override void PerformVoting(VotingEntry votingEntry)
        {
            if (votingEntry.Entropy > 1)
            {
                votingEntry.Label = SentimentLabel.Neutral;
            }
            else
            {
                base.PerformVoting(votingEntry);
            }
        }
    }
}