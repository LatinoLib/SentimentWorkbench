using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using Amib.Threading;
using Latino;
using Latino.Model;
using Latino.Model.Eval;
using Latino.TextMining;
using TextTask.DataSource;
using Action = System.Action;

namespace TextTask
{
    public abstract class Task 
    {
        protected Task()
        {
            Reports = new List<TaskReport>();
            BeforeActionPipe = new ActionPipe();
            AfterActionPipe = new ActionPipe();
            WorkItemPriority = WorkItemPriority.Normal;
        }

        public string ExperimentName { get; set; }
        public virtual string Name { get { return GetType().Name; } }
        public List<TaskReport> Reports { get; private set; }
        public TaskReport[] ErrorReports
        {
            get { return Reports.Where(r => r is ErrorReport).ToArray(); }
        }
        public bool AbortOnError { get; set; }
        public ActionPipe BeforeActionPipe { get; private set; }
        public ActionPipe AfterActionPipe { get; private set; }

        public TaskContext Context { get; set; }
        public System.Func<LabeledTextSource> DataSourceFactory { get; set; }
        public System.Func<TaskContext> ContextFactory { get; set; }

        public SmartThreadPool ThreadPool { get; set; }
        public WorkItemPriority WorkItemPriority { get; set; }

        public DateTime? PerformStart { get; private set; }
        public DateTime? PerformEnd { get; private set; }
        public TimeSpan? PerformDuration { get; private set; }
        public PerfData<SentimentLabel> PerfData { get; set; }
        public Dictionary<string, double> PerfValues { get; set; }

        public Task WithReport(TaskReport report)
        {
            Reports.Add(report);
            return this;
        }

        public Task WithReports(IEnumerable<TaskReport> reports)
        {
            Reports.AddRange(reports);
            return this;
        }

        public Task WithThreadPool(SmartThreadPool threadPool)
        {
            ThreadPool = threadPool;
            return this;
        }

        public Task WithPriority(WorkItemPriority priority)
        {
            WorkItemPriority = priority;
            return this;
        }

        public Task WithContext(TaskContext context)
        {
            Context = context;
            return this;
        }

        public Task PipeBefore(Action action)
        {
            BeforeActionPipe.Pipe(action);
            return this;
        }

        public Task PipeAfter(Action action)
        {
            AfterActionPipe.Pipe(action);
            return this;
        }

        public T Cast<T>() where T : Task
        {
            return (T)this;
        }

        public ActionPipe GetActionPipe()
        {
            var mainPipe = new ActionPipe
                {
                    ThreadPool = ThreadPool,
                    WorkItemPriority = WorkItemPriority,
                    OnException = OnException
                };
            PrepareActionPipe(mainPipe);

            var taskPipe = new ActionPipe
                {
                    ThreadPool = mainPipe.ThreadPool,
                    WorkItemPriority = mainPipe.WorkItemPriority,
                    OnException = mainPipe.OnException
                };
            taskPipe.Pipe(() => PerformStart = DateTime.Now);
            taskPipe.Pipe(BeforeActionPipe);
            taskPipe.Pipe(mainPipe);
            taskPipe.Pipe(() =>
            {
                PerformEnd = DateTime.Now;
                PerformDuration = PerformEnd - PerformStart;
            });
            taskPipe.Pipe(MakeReport);
            taskPipe.Pipe(AfterActionPipe);

            return taskPipe;
        }

        public Task StartExecution()
        {
            Exec(false);
            return this;
        }

        public Task Execute()
        {
            Exec(true);            
            return this;
        }

        public void MakeErrorReport(Exception e)
        {
            foreach (ErrorReport report in ErrorReports)
            {
                report.Exception = e;
                report.Make(this);
            }
        }

        private void OnException(Exception e, out bool abort)
        {
            MakeErrorReport(e);
            abort = AbortOnError;
        }

        private void Exec(bool inSync)
        {
            try
            {
                PerformStart = DateTime.Now;
                try
                {
                    ActionPipe pipe = GetActionPipe();
                    if (pipe != null)
                    {
                        if (inSync) { pipe.Execute(); } else { pipe.StartExecution(); }
                    }
                }
                finally
                {
                    PerformEnd = DateTime.Now;
                    PerformDuration = PerformEnd - PerformStart;
                }
            }
            catch (Exception e)
            {
                MakeErrorReport(e);
            }
        }

        private void MakeReport()
        {
            foreach (TaskReport report in Reports.Where(r => !(r is ErrorReport)))
            {
                report.Make(this);
            }
        }

        protected abstract void PrepareActionPipe(ActionPipe actionPipe);
    }


    public class SimpleTask : Task
    {
        private readonly string mName;

        public SimpleTask(string name = null) 
        {
            mName = name;
        }

        public Action OnExecute { get; set; }

        public override string Name
        {
            get { return mName; }
        }

        protected override void PrepareActionPipe(ActionPipe actionPipe)
        {
            actionPipe.Pipe(Preconditions.CheckNotNull(OnExecute));
        }
    }


    public static class TaskUtils
    {
        public static string GetConnectionString(string connectionStringId = "default")
        {
            var conSett = ConfigurationManager.ConnectionStrings[connectionStringId];
            return conSett == null ? null : conSett.ConnectionString;
        }

        public static LabeledDataset<SentimentLabel, SparseVector<double>> InitBowSpace(BowSpace bowSpace, 
            IEnumerable<LabeledExample<SentimentLabel, string>> labeledExamples, IEnumerable<string> initExamples = null)
        {
            LabeledExample<SentimentLabel, string>[] examples = labeledExamples as LabeledExample<SentimentLabel, string>[] ?? labeledExamples.ToArray();

            List<SparseVector<double>> bowData;
            if (initExamples != null)
            {
                Preconditions.CheckArgument(!(bowSpace is DeltaBowSpace<SentimentLabel>));
                bowSpace.Initialize(initExamples);
                bowData = examples.Select(le => bowSpace.ProcessDocument(le.Example)).ToList();
            }
            else
            {
                bowData = bowSpace is DeltaBowSpace<SentimentLabel>
                    ? ((DeltaBowSpace<SentimentLabel>)bowSpace).Initialize(new LabeledDataset<SentimentLabel, string>(examples))
                    : bowSpace.Initialize(examples.Select(d => d.Example));
            }

            var bowDataset = new LabeledDataset<SentimentLabel, SparseVector<double>>();
            for (int i = 0; i < bowData.Count; i++)
            {
                bowDataset.Add(examples[i].Label, bowData[i]);
            }
            return bowDataset;
        }

        public static void ProcessFeatures(TaskContext taskContext, IEnumerable<LabeledExample<SentimentLabel, string>> labeledExamples)
        {
            if (taskContext.FeatureProcessor != null)
            {
                foreach (LabeledExample<SentimentLabel, string> le in labeledExamples)
                {
                    le.Example = taskContext.FeatureProcessor.Run(le.Example);
                }
            }
        }
    }


    public class BunchOfExceptions : Exception
    {
        public Exception[] Exceptions { get; private set; }

        public BunchOfExceptions(IEnumerable<Exception> exceptions)
        {
            Exceptions = exceptions as Exception[] ?? exceptions.ToArray();
        }

        public BunchOfExceptions(string message, IEnumerable<Exception> exceptions)
            : base(message)
        {
            Exceptions = exceptions as Exception[] ?? exceptions.ToArray();
        }

        public BunchOfExceptions(string message, Exception innerException, IEnumerable<Exception> exceptions)
            : base(message, innerException)
        {
            Exceptions = exceptions as Exception[] ?? exceptions.ToArray();
        }
    }
}