using System;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public class ThreePlaneOneVsOneVotingClassifier : VotingClassifier<SentimentLabel, SparseVector<double>>
    {
        public ThreePlaneOneVsOneVotingClassifier() : base(new IModel<SentimentLabel, SparseVector<double>>[3])
        {
        }

        public ThreePlaneOneVsOneVotingClassifier(IModel<SentimentLabel, SparseVector<double>>[] innerModels) : base(innerModels)
        {
            Preconditions.CheckNotNull(innerModels);
            Preconditions.CheckArgument(innerModels.Length == 3);
        }

        public ThreePlaneOneVsOneVotingClassifier(BinarySerializer reader) : base(reader)
        {
        }

        protected override LabeledDataset<SentimentLabel, SparseVector<double>> GetTrainSet(int modelIdx, IModel<SentimentLabel, 
            SparseVector<double>> model, LabeledDataset<SentimentLabel, SparseVector<double>> trainSet)
        {
            switch (modelIdx)
            {
                case 0: return new LabeledDataset<SentimentLabel, SparseVector<double>>(trainSet.Where(le => le.Label != SentimentLabel.Neutral));
                case 1: return new LabeledDataset<SentimentLabel, SparseVector<double>>(trainSet.Where(le => le.Label != SentimentLabel.Negative));
                case 2: return new LabeledDataset<SentimentLabel, SparseVector<double>>(trainSet.Where(le => le.Label != SentimentLabel.Positive));
                default: throw new Exception();
            }
        }

        protected override void PerformVoting(VotingEntry votingEntry)
        {
            if (votingEntry.LabelProbs[SentimentLabel.Neutral] == votingEntry.LabelProbs[SentimentLabel.Negative] && 
                votingEntry.LabelProbs[SentimentLabel.Negative] == votingEntry.LabelProbs[SentimentLabel.Positive])
            //if (votingEntry.Entropy > 1)
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