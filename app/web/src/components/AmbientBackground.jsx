import React from 'react'
import './AmbientBackground.css'

export default function AmbientBackground() {
  return (
    <div className="ambient" aria-hidden="true">
      <div className="ambient__stars"></div>
      <div className="ambient__gradient"></div>
      <div className="ambient__grid"></div>
      <div className="ambient__scan"></div>
      <div className="ambient__glimmer"></div>
      <div className="ambient__orb ambient__orb--1"></div>
      <div className="ambient__orb ambient__orb--2"></div>
      <div className="ambient__orb ambient__orb--3"></div>
    </div>
  )
}
