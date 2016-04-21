using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.Model.Eval;

namespace TextTask.General
{
    public class ValidationTask : Task // todo delta bow space
    {
        public ValidationTask(TaskContext taskContext) 
        {
            Context = Preconditions.CheckNotNull(taskContext);

            LabeledExample<SentimentLabel, string>[] labeledExamples = taskContext.DataSource.GetData().ToArray();
            TaskUtils.ProcessFeatures(taskContext, labeledExamples);
            var labeledDataset = new LabeledDataset<SentimentLabel, string>(labeledExamples);

            // lazy model creation
            IEnumerable<Func<IModel<SentimentLabel, SparseVector<double>>>> modelFacotry = Enumerable.Range(0, taskContext.Models.Length)
                .Select<int, Func<IModel<SentimentLabel, SparseVector<double>>>>(i => () => taskContext.ModelFactory(i));

            Validator = new FoldLocalBowCrossValidator<SentimentLabel>(modelFacotry)
                {
                    Dataset = labeledDataset,
                    BowSpaceFunc = taskContext.BowSpaceFactory,
                    ModelNameFunc = (sender, m) => taskContext.GetModelName(m)
                };
        }

        public TaskMappingCrossValidator<SentimentLabel, string, SparseVector<double>> Validator { get; private set; }

        public override string Name
        {
            get { return Validator == null ? "cross-validation" : string.Format("{0}-fold cross-validation", Validator.NumFolds); }
        }

        protected override void PrepareActionPipe(ActionPipe actionPipe)
        {
            actionPipe.Join(Validator.GetFoldTasks());

            Validator.ExpName = ExperimentName;
            //Action matrixTask = () => ModelPerfMatrices = Validator.Models
                //.Select(m => Validator.PerfData.GetSumPerfMatrix(Validator.ExpName, Validator.GetModelName(m))).ToArray();
            //actionPipe.Pipe(matrixTask);
        }
    }


    public class UpfrontBowValidatorTask : Task
    {
        public UpfrontBowValidatorTask(TaskContext taskContext)
        {
            Context = Preconditions.CheckNotNull(taskContext);

            LabeledExample<SentimentLabel, string>[] labeledExamples = taskContext.DataSource.GetData().ToArray();
            TaskUtils.ProcessFeatures(taskContext, labeledExamples);

            // lazy model creation
            IEnumerable<Func<IModel<SentimentLabel, SparseVector<double>>>> modelFacotry = Enumerable.Range(0, taskContext.Models.Length)
                .Select<int, Func<IModel<SentimentLabel, SparseVector<double>>>>(i => () => taskContext.ModelFactory(i));

            Validator = new TaskCrossValidator<SentimentLabel, SparseVector<double>>(modelFacotry)
            {
                Dataset = TaskUtils.InitBowSpace(taskContext.BowSpace, labeledExamples),
                ModelNameFunc = (sender, m) => taskContext.GetModelName(m)
            };
        }

        public TaskCrossValidator<SentimentLabel, SparseVector<double>> Validator { get; private set; }

        public override string Name
        {
            get
            {
                return Validator == null
                    ? "upfront bow-space cross-validation"
                    : string.Format("{0}-fold upfront bow-space cross-validation", Validator.NumFolds);
            }
        }

        protected override void PrepareActionPipe(ActionPipe actionPipe)
        {
            actionPipe.Join(Validator.GetFoldTasks());

            Validator.ExpName = ExperimentName;
            //Action matrixTask = () => ModelPerfMatrices = Validator.Models
                //.Select(m => Validator.PerfData.GetSumPerfMatrix(Validator.ExpName, Validator.GetModelName(m))).ToArray();
            //actionPipe.Pipe(matrixTask);
        }
    }
}