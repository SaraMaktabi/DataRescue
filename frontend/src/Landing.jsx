import React from 'react'
import '@fontsource-variable/manrope/wght.css'
import prevynSymbol from './assets/prevyn-symbol.png'
import prevynWordmark from './assets/prevyn-wordmark.png'
import PipelineFlow from './PipelineFlow'
import './landing.css'

export default function Landing() {
  return <div className="landing">
    <header className="landing-header">
      <a className="landing-brand" href="#accueil" aria-label="Prevyn, accueil">
        <img className="landing-brand-symbol" src={prevynSymbol} alt="" width="386" height="300" />
        <img className="landing-brand-wordmark" src={prevynWordmark} alt="" width="295" height="153" />
      </a>
      <nav className="landing-links" aria-label="Navigation principale">
        <a className="landing-nav" href="#pipeline">Pipeline</a>
        <a className="landing-nav" href="#supervision">Supervision</a>
      </nav>
    </header>
    <main className="landing-main">
      <PipelineFlow />
      <div className="landing-copy">
        <p className="landing-kicker">Supervision des pipelines de données</p>
        <h1>Suivez vos pipelines.<br /><span>Anticipez les pannes.</span></h1>
        <p className="landing-summary">Explorez les alertes de la clôture bancaire, comprenez les facteurs de risque et comparez les prédictions aux résultats simulés.</p>
        <a className="landing-action" href="#supervision">Accéder au tableau de bord</a>
      </div>
    </main>
    <section className="pipeline-feature" id="pipeline" aria-labelledby="pipeline-feature-title">
      <div className="pipeline-window">
        <div className="pipeline-window-top" aria-hidden="true"><span>○ ○ ○</span><span>✦</span></div>
        <div className="pipeline-glass" aria-hidden="true">
          {Array.from({ length: 13 }, (_, index) => <i key={index} style={{ '--angle': `${-90 + index * 15}deg` }} />)}
        </div>
        <div className="pipeline-window-copy">
          <h2 id="pipeline-feature-title">De la donnée brute<br />à l’alerte.</h2>
          <p>La clôture bancaire ingère les données, les transforme et contrôle les résultats. Le prédicteur estime le risque de panne pour chaque tâche.</p>
          <ul aria-label="Étapes du pipeline"><li>Bronze</li><li>Silver</li><li>Gold</li></ul>
        </div>
      </div>
    </section>
    <section className="alert-feature" aria-label="Analyse des alertes">
      <a href="#supervision" aria-label="Repérez le signal. Comprenez l’alerte. Explorer les alertes.">
        <img src="/alert-prevyn-reference.png" alt="" loading="lazy" width="1672" height="941" />
      </a>
    </section>
  </div>
}
