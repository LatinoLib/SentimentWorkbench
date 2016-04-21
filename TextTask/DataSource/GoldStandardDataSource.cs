using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.TextMining;

namespace TextTask.DataSource
{
    public enum GoldStandardLabelKind
    {
        LabelWgtPrefNoSentiment, // Based on weights, preference towards exclude and neutral
        LabelWgtPrefSentiment, // Based on weights, preference towards sentiment
        LabelCountPrefNoSentiment, // Based on counts, preference towards exclude and neutral
        LabelCountPrefSentiment // Based on counts, preference towards sentiment
    }

    public class GoldStandardDataConfig
    {
        public GoldStandardDataConfig()
        {
            LabelKind = GoldStandardLabelKind.LabelCountPrefSentiment;
            IgnoreExcludes = true;
        }

        public GoldStandardLabelKind LabelKind { get; set; }
        public string[] DomainNames { get; set; }
        public string[] ProjectNames { get; set; }
        public string[] InclEntitiesDisj { get; set; }
        public string[] InclEntitiesConj { get; set; }
        public string[] ExclEntities { get; set; }
        public string[] Languages { get; set; }
        public bool IgnoreExcludes { get; set; }
        public int MaxDuplicateCount { get; set; }

        public GoldStandardDataConfig Clone()
        {
            return (GoldStandardDataConfig)MemberwiseClone();
        }

        public override string ToString()
        {
            string domains = string.Join(" ", DomainNames ?? new[] { "" });
            string projects = string.Join(" ", ProjectNames ?? new[] { "" });
            string inclEntities = string.Join(" ", InclEntitiesDisj ?? new[] { "" });
            string andInclEntities = string.Join(" ", InclEntitiesConj ?? new[] { "" });
            if (!string.IsNullOrEmpty(andInclEntities)) { andInclEntities = "&(" + andInclEntities + ")"; }
            string exclEntities = string.Join(" ", ExclEntities ?? new[] { "" });
            if (!string.IsNullOrEmpty(exclEntities)) { exclEntities = "!(" + exclEntities + ")"; }
            string langs = string.Join(" ", Languages ?? new[] { "" });
            return string.Join(" ", new[] { domains, projects, inclEntities, andInclEntities, exclEntities, langs });
        }
    }

    public class GoldStandardDataSource : DatabaseLabeledTextSource
    {
        private static readonly Dictionary<string, Language> mStringLanguages = new Dictionary<string, Language> 
            { { "en", Language.English }, { "de", Language.German } };

        private IEnumerable<LabeledExample<SentimentLabel, Tweet>> mData;
        private int? mDataSize;
        private GoldStandardDataConfig mConfig;

        public GoldStandardDataSource(string connectionString) : base(connectionString)
        {
        }

        public GoldStandardDataConfig Config
        {
            get { return mConfig; }
            set
            {
                mConfig = value;
                if (mConfig != null && mConfig.Languages != null && mConfig.Languages.Length == 1)
                {
                    Language = mStringLanguages[Config.Languages[0]];
                }
                else
                {
                    Language = Language.Unspecified;
                }
            }
        }

        public override int DataSize
        {
            get
            {
                if (mDataSize == null)
                {
                    if (mData == null) { return 0; }
                    mDataSize = mData.Count();
                }
                return mDataSize.Value;
            }
        }

        public override IEnumerable<LabeledExample<SentimentLabel, string>> GetData()
        {
            return GetData<Tweet>().Select(lt => new LabeledExample<SentimentLabel, string>(lt.Label, lt.Example.Text));
        }

        public override IEnumerable<LabeledExample<SentimentLabel, T>> GetData<T>() 
        {
            Preconditions.CheckArgument(typeof(T) == typeof(Tweet));
            Preconditions.CheckNotNull(Config);
            Preconditions.CheckArgumentRange(DataShare > 0 && DataShare <= 1);

            if (mData != null)
            {
                return mData.Cast<LabeledExample<SentimentLabel, T>>();
            }

            string domainsTerm = Config.DomainNames == null || !Config.DomainNames.Any() ? ""
                : string.Format(" AND d.Name in ({0})", string.Join(",", Config.DomainNames.Select(n => "'" + n + "'")));

            string projectsTerm = Config.ProjectNames == null || !Config.ProjectNames.Any() ? ""
                : string.Format(" AND p.IdStr in ({0})", string.Join(",", Config.ProjectNames.Select(n => "'" + n + "'")));

            string entitiesTerm = Config.InclEntitiesDisj == null || !Config.InclEntitiesDisj.Any() ? ""
                : string.Format(" AND ({0})", string.Join(" OR ", Config.InclEntitiesDisj.Select(n => string.Format("gs.ObjectId LIKE '{0}'", n))));

            if (Config.InclEntitiesConj != null && Config.InclEntitiesConj.Any())
            {
                entitiesTerm += string.Format(" AND ({0})", string.Join(" AND ", Config.InclEntitiesConj.Select(n => string.Format("gs.ObjectId LIKE '{0}'", n))));
            }
            if (Config.ExclEntities != null && Config.ExclEntities.Any())
            {
                entitiesTerm += string.Format(" AND ({0})", string.Join(" AND ", Config.ExclEntities.Select(n => string.Format("gs.ObjectId NOT LIKE '{0}'", n))));
            }

            string langsTerm = Config.Languages == null || !Config.Languages.Any() ? ""
                : string.Format(" AND gs.Language in ({0})", string.Join(",", Config.Languages.Select(n => "'" + n + "'")));

            string duplicateTerm = Config.MaxDuplicateCount > 0 ? "AND gs.DuplicateCount <= " + Config.MaxDuplicateCount : "";

            string sqlFromClause = string.Format(@"
				FROM GoldStandard gs
                  JOIN focus f
                         ON f.id = gs.focusid 
                       JOIN project p
                         ON p.id = f.ProjectId {1}
                      JOIN domainproject dp
                         ON p.id = dp.Project_Id 
                      JOIN domain d
                         ON d.id = dp.Domain_Id {2}
                WHERE gs.CreatedAt >= @p0 AND gs.CreatedAt < @p1 {4} {0} {3}
            ", entitiesTerm, projectsTerm, domainsTerm, langsTerm, duplicateTerm);

            int topCount = 0;
            if (MaxDataSize > 0)
            {
                topCount = MaxDataSize;
            }
            else if (DataShare < 1)
            {
                topCount = DbContext.Database.SqlQuery<int>("SELECT COUNT(*) " + sqlFromClause, From, To).Single();
                topCount = (int)Math.Truncate(DataShare * topCount);
            }
            string topTerm = topCount > 0 ? "TOP " + topCount : "";

            string orderTerm = " ORDER BY RAND() ";

            string sqlSelectClause = string.Format(@"
                SELECT {0} gs.Text, gs.CreatedAt Date,
                   CASE gs.{1}
                     WHEN 'Negative' THEN -1 
                     WHEN 'Neutral' THEN 0 
                     WHEN 'Positive' THEN 1 
                     WHEN 'Exclude' THEN 10
                   END Label, 
                   CASE gs.countnegative * gs.countpositive 
                     WHEN 0 THEN 
                       CASE gs.countnegative * gs.countneutral + gs.countneutral * gs.countpositive 
                         WHEN 0 THEN 0 
                         ELSE 1 
                       END 
                     ELSE 2 
                   END DisagreementRank,
                   CASE gs.countnegative * gs.countpositive 
                     WHEN 0 THEN 
                       CASE gs.countnegative * gs.countneutral + gs.countneutral * gs.countpositive 
                         WHEN 0 THEN gs.countnegative + gs.countneutral + gs.countpositive 
                         ELSE 0
                       END 
                     ELSE 0
                   END AgreementRank 
            ", topTerm, Config.LabelKind);

            IEnumerable<LabeledTweet> labeledTweets = DbContext.Database
                .SqlQuery<LabeledTweet>(sqlSelectClause + sqlFromClause + orderTerm, From, To)
                .Where(lt => lt.Label != null);

            mData = Config.IgnoreExcludes
                ? labeledTweets.Where(lt => lt.Label != SentimentLabel.Exclude)
                    .Select(lt => new LabeledExample<SentimentLabel, Tweet>(lt.Label, lt))
                : labeledTweets.Select(lt => new LabeledExample<SentimentLabel, Tweet>(lt.Label, lt));

            return mData.Cast<LabeledExample<SentimentLabel, T>>();
        }

        public override string Name { get { return ToString(); } }

        public override void Dispose()
        {
            mData = null;
            base.Dispose();
        }

        public override string ToString()
        {
            return Config == null ? "" : Config.ToString();
        }

        private class LabeledTweet : Tweet
        {
            public SentimentLabel Label { get; set; }
        }
    }
}