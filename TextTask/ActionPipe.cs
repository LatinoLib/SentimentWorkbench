using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Amib.Threading;
using Latino;
using Action = System.Action;

namespace TextTask
{
    public class ActionPipe : IEnumerable<IEnumerable<Action>>
    {
        private readonly List<List<Action>> mActionGroups = new List<List<Action>>();

        public enum ActionGrouppingKind
        {
            Pipe,
            Join
        }

        public ActionPipe(SmartThreadPool threadPool = null) 
        {
            ThreadPool = threadPool;
            WorkItemPriority = WorkItemPriority.Normal;
        }

        public ActionGrouppingKind GrouppingKind { get; set; }

        public WorkItemPriority WorkItemPriority { get; set; }
        public SmartThreadPool ThreadPool { get; set; }
        public delegate void ExceptionHandler(Exception e, out bool abort);
        public ExceptionHandler OnException { get; set; }
        public Exception ExecutionException { get; private set; }

        public ActionPipe Join()
        {
            GrouppingKind = ActionGrouppingKind.Join;
            return this;
        }

        public ActionPipe Join(IEnumerable<Action> actions)
        {
            Preconditions.CheckArgument(actions != null);
            mActionGroups.Add(new List<Action>());
            mActionGroups.Last().AddRange(actions);
            return Join();
        }

        public ActionPipe Pipe()
        {
            GrouppingKind = ActionGrouppingKind.Pipe;
            return this;
        }

        public ActionPipe Pipe(Action action)
        {
            Preconditions.CheckArgument(action != null);
            mActionGroups.Add(new List<Action> { action });
            return Pipe();
        }

        public ActionPipe Pipe(IEnumerable<Action> actions)
        {
            Preconditions.CheckArgument(actions != null);
            mActionGroups.AddRange(actions.Select(a => new List<Action> { a }));
            return Pipe();
        }

        public ActionPipe Pipe(ActionPipe actionPipe)
        {
            mActionGroups.AddRange(actionPipe.Select(ag => ag.ToList()));
            return Pipe();
        }

        public ActionPipe Add(Action action)
        {
            switch (GrouppingKind)
            {
                case ActionGrouppingKind.Join:
                {
                    if (!mActionGroups.Any()) { mActionGroups.Add(new List<Action>()); }
                    mActionGroups.Last().Add(action);
                    break;
                }
                case ActionGrouppingKind.Pipe:
                    return Pipe(action);
            }
            return this;
        }

        public ActionPipe Add(IEnumerable<Action> actions)
        {
            switch (GrouppingKind)
            {
                case ActionGrouppingKind.Join:
                    return Join(actions);
                case ActionGrouppingKind.Pipe:
                    return Pipe(actions);
            }
            return this;
        }

        public IEnumerator<IEnumerable<Action>> GetEnumerator()
        {
            return mActionGroups.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void Execute()
        {
            if (ThreadPool == null)
            {
                StartExecution();
            }
            else
            {
                var finishEvent = new ManualResetEvent(false);
                StartExecution(finishEvent);
                finishEvent.WaitOne();
                finishEvent.Close();
                if (ExecutionException != null)
                {
                    throw new Exception("An error has occurred during execution. Check inner exception for details.", ExecutionException);
                }
            }
        }

        public void StartExecution(EventWaitHandle wainHandle = null)
        {
            ExecutionException = null;
            if (ThreadPool == null)
            {
                foreach (Action action in mActionGroups.SelectMany(ag => ag))
                {
                    action();
                }
            }
            else
            {
                List<List<Action>> actionGroups = mActionGroups.ToList();
                if (wainHandle != null)
                {
                    actionGroups.Add(new List<Action> { 
                        () => { try { wainHandle.Set(); } catch { } } 
                    });
                }

                IWorkItemsGroup[] wigs = actionGroups.Select(ag => ThreadPool.CreateWorkItemsGroup(ag.Count, new WIGStartInfo
                    {
                        WorkItemPriority = WorkItemPriority
                    })).ToArray();

                // make wig chain via on-idle event
                for (int i = wigs.Length - 2; i >= 0; i--)
                {
                    int j = i + 1;
                    wigs[i].OnIdle += g =>
                    {
                        if (ExecutionException == null) { QueueWorkItems(wigs[j], actionGroups[j]); }
                        else
                        {
                            if (wainHandle != null) { try { wainHandle.Set(); } catch { } }
                        }
                    };
                }
                // start off by running first member of the chain
                QueueWorkItems(wigs[0], actionGroups[0]);
            }
        }

        public static void WaitAll(ICollection<WaitHandle> waitHandles)
        {
            if (Preconditions.CheckNotNull(waitHandles).All(h => h == null)) { return; }

            // workaround for limitation of WaitAll with max 64 handles
            for (int i = 0; i <= (waitHandles.Count(h => h != null) - 1) / 64; i++)
            {
                WaitHandle.WaitAll(waitHandles.Skip(i * 64).Take(64).ToArray());
            }
        }

        public static void CloseAll(ICollection<WaitHandle> waitHandles)
        {
            if (Preconditions.CheckNotNull(waitHandles).All(h => h == null)) { return; }

            foreach (WaitHandle waitHandle in waitHandles.Where(h => h != null))
            {
                waitHandle.Close();
            }
        }

        private void QueueWorkItems(IWorkItemsGroup group, IEnumerable<Action> actionGroup)
        {
            foreach (Action action in actionGroup)
            {
                Action action_ = action;
                group.QueueWorkItem(
                    o =>
                    {
                        try
                        {
                            action_();
                        }
                        catch (Exception e)
                        {
                            ExecutionException = e;
                            bool abort = false;
                            if (OnException != null) { OnException(e, out abort); }
                            group.Cancel(abort);
                        }
                        return null;
                    },
                    this,
                    wi =>
                    {
                        //if (wi.Exception != null) { }
                    });
            }
        }
    }
}