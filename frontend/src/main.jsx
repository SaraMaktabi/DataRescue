import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import Landing from './Landing'
import './style.css'

const outcomes = {
  detected: { label: 'Panne détectée', tone: 'mint' },
  false: { label: 'Fausse alerte', tone: 'amber' },
  missed: { label: 'Panne manquée', tone: 'rose' },
  success: { label: 'Succès', tone: 'muted' },
}
const taskNames = {
  bronze_ingest: 'Transactions', accounts_ingest: 'Comptes', fx_ingest: 'Taux de change',
  silver_clean: 'Nettoyage', pseudonymize: 'Pseudonymisation', join_accounts: 'Jointure des comptes',
  reconcile: 'Réconciliation', compute_balances: 'Calcul des soldes',
  aggregate_risk: 'Agrégation du risque', export_reporting: 'Export du rapport',
}
const stages = [
  { name: 'Bronze', label: 'Ingestion', tasks: ['bronze_ingest', 'accounts_ingest', 'fx_ingest'] },
  { name: 'Silver', label: 'Transformation', tasks: ['silver_clean', 'pseudonymize', 'join_accounts'] },
  { name: 'Gold', label: 'Contrôle et restitution', tasks: ['reconcile', 'compute_balances', 'aggregate_risk', 'export_reporting'] },
]
const factors = {
  input_null_rate_pct: 'Taux de valeurs manquantes',
  volume_delta_pct: 'Écart de volume',
  input_rows: 'Nombre de lignes',
  volume_mean_7d: 'Volume moyen sur 7 jours',
  day_of_week: 'Jour de la semaine',
  task_id_reconcile: 'Identité de la tâche',
}
const number = new Intl.NumberFormat('fr-FR')
const percent = new Intl.NumberFormat('fr-FR', { style: 'percent', maximumFractionDigits: 1 })
const score = new Intl.NumberFormat('fr-FR', { minimumFractionDigits: 3, maximumFractionDigits: 3 })
const date = new Intl.DateTimeFormat('fr-FR', { day: 'numeric', month: 'short', timeZone: 'UTC' })
const outcome = row => row.alert ? (row.failed ? 'detected' : 'false') : (row.failed ? 'missed' : 'success')
const pageSize = 8

function Badge({ row }) {
  const { label, tone } = outcomes[outcome(row)]
  return <span className={`badge ${tone}`}>{label}</span>
}

function Activity({ predictions }) {
  const days = new Map()
  for (const row of predictions) {
    const totals = days.get(row.date) ?? { alerts: 0, failures: 0 }
    totals.alerts += Number(row.alert)
    totals.failures += Number(row.failed)
    days.set(row.date, totals)
  }
  const series = [...days.entries()].sort(([first], [second]) => first.localeCompare(second))
  const weeks = []
  for (let start = 0; start < series.length; start += 7) {
    const slice = series.slice(start, start + 7)
    weeks.push(slice.reduce((total, [, day]) => ({
      ...total, alerts: total.alerts + day.alerts, failures: total.failures + day.failures,
    }), { start: slice[0][0], end: slice.at(-1)[0], alerts: 0, failures: 0 }))
  }
  const maximum = Math.max(1, ...weeks.flatMap(week => [week.alerts, week.failures]))
  return <article className="panel activity">
    <h2>Alertes et pannes</h2>
    <p className="activity-legend"><span>Alertes émises</span><span>Pannes simulées</span></p>
    <ul className="activity-chart" aria-label="Alertes et pannes par période de sept jours">
      {weeks.map(week => {
        const label = `${date.format(new Date(week.start))} au ${date.format(new Date(week.end))} · ${week.alerts} alertes, ${week.failures} pannes`
        return <li key={week.start} title={label} aria-label={label}>
          <span style={{ '--value': `${week.alerts / maximum * 100}%` }} />
          <span style={{ '--value': `${week.failures / maximum * 100}%` }} />
        </li>
      })}
    </ul>
  </article>
}

function ModelCard({ threshold, metrics, predictions }) {
  return <section className="model-overview" aria-label="Modèle actif et résultats">
    <article className="bank-card" aria-label="Carte du modèle XGBoost">
      <div className="bank-card-top">
        <strong>Prevyn</strong>
        <span className="bank-chip" aria-hidden="true" />
      </div>
      <div className="bank-card-model">
        <span>Modèle actif</span>
        <h2>XGBoost</h2>
      </div>
      <div className="bank-card-bottom">
        <span>Seuil d’alerte <strong>{score.format(threshold)}</strong></span>
        <span className="card-signal" aria-hidden="true"><i /><i /><i /></span>
      </div>
    </article>
    <article className="panel model-performance">
      <div className="recall-dial" style={{ '--recall': `${(metrics.recall ?? 0) * 100}%` }}>
        <strong>{percent.format(metrics.recall)}</strong>
      </div>
      <div>
        <h2>Rappel des pannes</h2>
        <p>{metrics.true_positives} pannes détectées sur {metrics.true_positives + metrics.false_negatives}</p>
        <p className="accuracy">Accuracy globale <strong>{percent.format(metrics.accuracy)}</strong></p>
      </div>
    </article>
    <Activity predictions={predictions} />
  </section>
}

function Detail({ row, threshold, explanation, close, panelRef }) {
  if (!row) return <aside className="panel detail empty-detail">Sélectionne une exécution pour voir son détail.</aside>
  return (
    <aside className="panel detail" aria-labelledby="detail-title" ref={panelRef} tabIndex={-1}>
      <div className="section-heading">
        <span className="eyebrow">Détail de l’exécution</span>
        <button className="close" aria-label="Fermer le détail" onClick={close}>×</button>
      </div>
      <h2 id="detail-title">{taskNames[row.task] ?? row.task}</h2>
      <p className="run-id">{row.run_id}</p>
      <Badge row={row} />
      <div className="timeline">
        <div><span className="eyebrow">Avant · score XGBoost</span><strong className="detail-score">{score.format(row.score)}</strong>
          <div className="risk-meter" style={{ '--threshold': `${threshold * 100}%` }}>
            <meter min="0" max="1" value={row.score} aria-label="Score de risque" />
          </div>
          <small>Seuil {score.format(threshold)} · {row.alert ? 'Alerte émise' : 'Aucune alerte'}</small>
        </div>
        <div><span className="eyebrow">Après · résultat simulé</span><strong>{row.failed ? 'Tâche en panne' : 'Tâche réussie'}</strong></div>
      </div>
      <h3>Facteurs SHAP</h3>
      {explanation?.length ? (
        <ol className="factors">{explanation.map(item => {
          const match = /^(.*) \(([^)]+)\)$/.exec(item)
          return <li key={item}><span>{factors[match?.[1]] ?? match?.[1] ?? item}</span>{match && <code>{match[2]}</code>}</li>
        })}</ol>
      ) : <p className="muted-text">Aucune explication enregistrée pour cette exécution.</p>}
    </aside>
  )
}

function Comparison({ models }) {
  const names = { random_forest: 'Random Forest', xgboost: 'XGBoost', isolation_forest: 'Isolation Forest' }
  const maxCount = Math.max(1, ...models.flatMap(model => ['true_positives', 'false_negatives', 'false_positives'].map(key => Number(model[key]))))
  return <section className="panel comparison">
    <h2>Comparaison des modèles</h2>
    <p className="muted-text">Même période de test et même budget d’alertes en validation.</p>
    {!models.length && <p>Le tableau comparatif est indisponible.</p>}
    <div className="model-grid">{models.map(model => <article className="model" key={model.model}>
      <h3>{names[model.model] ?? model.model}</h3>
      {[
        ['Pannes détectées', 'true_positives', 'mint'],
        ['Pannes manquées', 'false_negatives', 'rose'],
        ['Fausses alertes', 'false_positives', 'amber'],
      ].map(([label, key, tone]) => <div className="bar-item" key={key}>
        <div><span>{label}</span><strong>{model[key]}</strong></div>
        <div className="track" aria-hidden="true"><span className={tone} style={{ width: `${Number(model[key]) / maxCount * 100}%` }} /></div>
      </div>)}
      <footer>Rappel <strong>{percent.format(model.recall)}</strong> · Accuracy <strong>{percent.format(model.accuracy)}</strong></footer>
    </article>)}</div>
  </section>
}

function Dashboard({ data }) {
  const [view, setView] = useState('supervision')
  const [task, setTask] = useState('')
  const [filter, setFilter] = useState('')
  const [day, setDay] = useState('')
  const [sortBy, setSortBy] = useState('date')
  const [page, setPage] = useState(0)
  const detailRef = useRef(null)
  const [selected, setSelected] = useState(() =>
    data.predictions.find(row => outcome(row) === 'false' && data.explanations[row.run_id]) ?? null,
  )
  const contextualRows = useMemo(() => data.predictions.filter(row =>
    (!day || row.date === day) && (!filter || outcome(row) === filter),
  ), [data.predictions, day, filter])
  const counts = useMemo(() => contextualRows.reduce((acc, row) => {
    acc[row.task] = (acc[row.task] ?? 0) + 1
    return acc
  }, {}), [contextualRows])
  const visible = contextualRows.filter(row => !task || row.task === task)
    .sort((a, b) => sortBy === 'score'
      ? b.score - a.score || b.date.localeCompare(a.date)
      : b.date.localeCompare(a.date) || b.score - a.score)
  const pages = Math.max(1, Math.ceil(visible.length / pageSize))
  const dates = [...new Set(data.predictions.map(row => row.date))].sort().reverse()
  const change = (setter, value) => {
    setter(value)
    setPage(0)
    setSelected(null)
  }
  const resetFilters = () => {
    setTask('')
    setFilter('')
    change(setDay, '')
  }
  const openDetail = row => {
    setSelected(row)
    requestAnimationFrame(() => {
      const panel = detailRef.current
      panel?.focus({ preventScroll: true })
      if (window.matchMedia('(max-width: 1020px)').matches) panel?.scrollIntoView({ block: 'nearest' })
    })
  }

  return <div className="app">
    <aside className="dashboard-sidebar" aria-label="Navigation Prevyn">
      <nav aria-label="Tableau de bord">
        <a href="#accueil"><span aria-hidden="true">⌂</span><span>Accueil</span></a>
        <button aria-current={view === 'supervision' ? 'page' : undefined} onClick={() => setView('supervision')}><span aria-hidden="true">▥</span><span>Supervision</span></button>
        <button aria-current={view === 'comparison' ? 'page' : undefined} onClick={() => setView('comparison')}><span aria-hidden="true">◫</span><span>Modèles</span></button>
      </nav>
    </aside>
    <main>
      <header className="intro">
        <div><p className="eyebrow">Clôture bancaire</p><h1>Supervision des pannes</h1></div>
        <div className="period"><span className="data-tag">Historique simulé</span><p>{date.format(new Date(data.period.start))} – {date.format(new Date(data.period.end))} {data.period.end.slice(0, 4)}<br /><span>{number.format(data.period.rows)} exécutions · XGBoost</span></p></div>
      </header>
      {view === 'comparison' ? <Comparison models={data.comparison} /> : <>
        <ModelCard threshold={data.threshold} metrics={data.metrics} predictions={data.predictions} />
        <section className="metrics" aria-label="Résultats du test">
          {[
            ['Pannes détectées', data.metrics.true_positives, 'mint', 'detected'],
            ['Pannes manquées', data.metrics.false_negatives, 'rose', 'missed'],
            ['Fausses alertes', data.metrics.false_positives, 'amber', 'false'],
          ].map(([label, value, tone, key]) => <button key={key}
            className={`panel metric ${tone}`} aria-pressed={filter === key}
            onClick={() => change(setFilter, filter === key ? '' : key)}>
            <span>{label}</span><strong>{value}</strong><span className="metric-action">{filter === key ? 'Retirer le filtre' : 'Filtrer les exécutions'}</span>
          </button>)}
        </section>
        <div className="workspace">
          <div className="primary">
            <section className="panel pipeline">
              <div className="section-heading"><h2>Pipeline de clôture</h2><span className="muted-text">Exécutions par tâche</span></div>
              <div className="stages">{stages.map(stage => <div className="stage" key={stage.name}>
                <h3>{stage.name}</h3>
                <div className="stage-folder">
                  <p>{stage.label}</p>
                  {stage.tasks.map(id => <button className="task" key={id} aria-pressed={task === id}
                  onClick={() => change(setTask, task === id ? '' : id)}>
                    <span>{taskNames[id]}</span><strong>{counts[id] ?? 0}</strong>
                  </button>)}
                </div>
              </div>)}</div>
            </section>
            <section className="panel runs">
              <div className="section-heading"><h2>Exécutions</h2><span className="muted-text" aria-live="polite">{number.format(visible.length)} résultats</span></div>
              <div className="filters">
                <label>Résultat<select value={filter} onChange={event => change(setFilter, event.target.value)}>
                  <option value="">Tous les résultats</option>
                  {Object.entries(outcomes).map(([key, option]) => <option key={key} value={key}>{option.label}</option>)}
                </select></label>
                <label>Tâche<select value={task} onChange={event => change(setTask, event.target.value)}>
                  <option value="">Toutes les tâches</option>
                  {Object.entries(taskNames).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
                </select></label>
                <label>Date<select value={day} onChange={event => change(setDay, event.target.value)}>
                  <option value="">Toutes les dates</option>
                  {dates.map(value => <option key={value}>{value}</option>)}
                </select></label>
                <label>Tri<select value={sortBy} onChange={event => change(setSortBy, event.target.value)}>
                  <option value="date">Plus récentes</option><option value="score">Risque le plus élevé</option>
                </select></label>
              </div>
              {(filter || task || day) && <button className="reset-filters" onClick={resetFilters}>Effacer les filtres</button>}
              {Object.keys(data.explanations).length > 0 && <div className="examples">
                <span>Alertes expliquées</span>
                {data.predictions.filter(row => data.explanations[row.run_id]).map(row =>
                  <button key={row.run_id} onClick={() => openDetail(row)}>{taskNames[row.task] ?? row.task}</button>)}
              </div>}
              <div className="table-scroll"><table>
                <thead><tr><th>Date</th><th>Tâche</th><th>Score de risque</th><th>Résultat de la prédiction</th></tr></thead>
                <tbody>{visible.slice(page * pageSize, (page + 1) * pageSize).map(row =>
                  <tr key={row.run_id} data-selected={selected?.run_id === row.run_id}>
                    <td>{date.format(new Date(row.date))}</td>
                    <td><button className="row-button" onClick={() => openDetail(row)}>{taskNames[row.task] ?? row.task}</button></td>
                    <td className="tabular">{score.format(row.score)}</td><td><Badge row={row} /></td>
                  </tr>)}
                  {!visible.length && <tr><td colSpan="4">Aucune exécution pour ces filtres.</td></tr>}
                </tbody>
              </table></div>
              <div className="pagination">
                <button disabled={page === 0} onClick={() => setPage(page - 1)}>Précédent</button>
                <span>Page {page + 1} / {pages}</span>
                <button disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Suivant</button>
              </div>
            </section>
          </div>
          <Detail row={selected} threshold={data.threshold} explanation={data.explanations[selected?.run_id]} close={() => setSelected(null)} panelRef={detailRef} />
        </div>
      </>}
    </main>
  </div>
}

function DashboardPage() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    fetch('/api/dashboard').then(async response => {
      if (!response.ok) throw new Error((await response.json()).detail || 'Chargement indisponible.')
      return response.json()
    }).then(setData).catch(reason => setError(reason.message))
  }, [])
  if (error) return <main className="notice"><h1>DataRescue</h1><p role="alert">{error}</p></main>
  if (!data) return <main className="notice"><p role="status">Chargement du tableau de bord…</p></main>
  return <Dashboard data={data} />
}

function App() {
  const [route, setRoute] = useState(window.location.hash)
  useEffect(() => {
    const navigate = () => {
      setRoute(window.location.hash)
      if (window.location.hash !== '#pipeline') window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', navigate)
    return () => window.removeEventListener('hashchange', navigate)
  }, [])
  useEffect(() => {
    if (route === '#pipeline') document.getElementById('pipeline')?.scrollIntoView()
  }, [route])
  return route === '#supervision' ? <DashboardPage /> : <Landing />
}

createRoot(document.getElementById('root')).render(<App />)
