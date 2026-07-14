# Parameter profiles

These JSON files hold parameter provenance, validity domains, nominal operating
points, and solver settings separately from executable equations. A profile is
metadata-only: loading it never changes `model.p`.

Profiles retain `legacy-unverified` until every relevant value has a traceable
source and a documented unit, valid range, and uncertainty where available.
`quadruple` is the first migrated `reference-parameterized` profile.
