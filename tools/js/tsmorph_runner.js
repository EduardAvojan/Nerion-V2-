#!/usr/bin/env node
// Minimal ts-morph runner for Nerion JS/TS transforms.
// Reads a JSON payload file: { source: string, actions: Array<...> }
// Writes JSON to stdout: { ok: boolean, source?: string, error?: string }

const fs = require('fs');

const CACHE = Object.create(null); // cacheKey -> { project }
let MESSAGES = [];

function fail(msg, ctx) {
  const out = { ok: false, error: String(msg) };
  if (ctx && typeof ctx === 'object') {
    if (ctx.action) out.action = ctx.action;
    if (ctx.index !== undefined) out.index = ctx.index;
    if (ctx.file) out.file = ctx.file;
    if (Array.isArray(MESSAGES) && MESSAGES.length) out.messages = MESSAGES;
  }
  process.stdout.write(JSON.stringify(out));
  process.exit(0);
}

async function main() {
  const payloadPath = process.argv[2];
  if (!payloadPath) return fail('missing payload path');
  let payload;
  try {
    payload = JSON.parse(fs.readFileSync(payloadPath, 'utf8'));
  } catch (e) {
    return fail('invalid payload');
  }
  const actions = Array.isArray(payload.actions) ? payload.actions : [];
  const files = Array.isArray(payload.files) ? payload.files : null;
  const primary = payload.primary || (files && files.length ? files[0].path : 'file.tsx');
  const cacheKey = payload.cacheKey || null;
  // Lazy load ts-morph
  let ts, Project, SyntaxKind;
  try {
    const tsm = require('ts-morph');
    Project = tsm.Project; SyntaxKind = tsm.SyntaxKind; ts = tsm;
  } catch (e) {
    return fail('ts-morph not available');
  }
  try {
    let project;
    if (cacheKey && CACHE[cacheKey]) {
      project = CACHE[cacheKey].project;
    } else {
      project = new Project({ useInMemoryFileSystem: true, manipulationSettings: { quoteKind: 'single' } });
      if (cacheKey) CACHE[cacheKey] = { project };
    }
    let sf;
    if (files) {
      for (const f of files) {
        const p = String(f.path || '').trim() || 'file.tsx';
        const s = String(f.source || '');
        const exist = project.getSourceFile(p);
        if (exist) exist.replaceWithText(s); else project.createSourceFile(p, s, { overwrite: true });
      }
      sf = project.getSourceFile(primary) || project.getSourceFiles()[0];
    } else {
      const source = String(payload.source || '');
      const exist = project.getSourceFile('file.tsx');
      if (exist) exist.replaceWithText(source); else project.createSourceFile('file.tsx', source, { overwrite: true });
      sf = project.getSourceFile('file.tsx');
    }

    const getImportDeclsFrom = (mod) => sf.getImportDeclarations().filter(d => (d.getModuleSpecifierValue && d.getModuleSpecifierValue() === mod));
    const getImportFrom = (mod) => {
      const list = getImportDeclsFrom(mod);
      return list.length ? list[0] : undefined;
    };
    const isSideEffectImport = (decl) => {
      try {
        return !decl.getDefaultImport() && !decl.getNamespaceImport() && decl.getNamedImports().length === 0;
      } catch { return false; }
    };
    const insertImportWithCluster = (opts, isSideEffect) => {
      const decls = sf.getImportDeclarations();
      let index = decls.length;
      if (decls.length) {
        if (isSideEffect) {
          // after last side-effect import, else after last import
          let lastSide = -1;
          for (let i = 0; i < decls.length; i++) if (isSideEffectImport(decls[i])) lastSide = i;
          index = lastSide >= 0 ? (lastSide + 1) : decls.length;
        } else {
          // before first side-effect import, else at end
          let firstSide = -1;
          for (let i = 0; i < decls.length; i++) if (isSideEffectImport(decls[i])) { firstSide = i; break; }
          index = firstSide >= 0 ? firstSide : decls.length;
        }
      }
      return sf.insertImportDeclaration(index, opts);
    };
    const ensureDefaultConflictFree = (mod) => {
      const decls = getImportDeclsFrom(mod);
      let defaults = new Set();
      for (const d of decls) {
        const di = d.getDefaultImport && d.getDefaultImport();
        if (di) defaults.add(di);
      }
      if (defaults.size > 1) {
        const names = Array.from(defaults).join(', ');
        // Add remediation suggestions
        MESSAGES.push({ level: 'error', code: 'IMPORT_DEFAULT_CONFLICT', suggestion: `Consider converting one default to named: import { ${names.split(', ')[0]} as Alias } from '${mod}'; or consolidate into a single default import.` });
        return `default import conflict for '${mod}': ${names}`;
      }
      return null;
    };

    const hasCommentsInNamed = (decl) => {
      try {
        const txt = decl.getText();
        const m = txt.match(/\{([\s\S]*?)\}/);
        if (!m) return false;
        const inside = m[1] || '';
        return inside.includes('/*') || inside.includes('//');
      } catch { return false; }
    };

    const addNamedPreserveComments = (decl, toAdd) => {
      try {
        const full = decl.getText();
        const m = full.match(/^(\s*import\s+[\s\S]+?\{)([\s\S]*?)(\})([\s\S]*?from\s+['\"][^'\"]+['\"];?)/);
        if (!m) return false;
        const before = m[1], inside = m[2], afterBrace = m[3], tail = m[4];
        let newInside = inside;
        for (const spec of toAdd) {
          const needle = spec.alias ? `${spec.name} as ${spec.alias}` : spec.name;
          if (newInside.includes(needle)) continue;
          const trimmed = newInside.trim();
          if (!trimmed) newInside = ` ${needle} `;
          else if (/[,\s]$/.test(newInside.trim())) newInside = `${newInside}${needle} `;
          else newInside = `${newInside}, ${needle} `;
        }
        const newText = `${before}${newInside}${afterBrace}${tail}`;
        decl.replaceWithText(newText);
        return true;
      } catch { return false; }
    };

    // --- tsconfig-aware resolution helpers ----------------------------------
    let tsconfig = null;
    try {
      const tc = fs.readFileSync('tsconfig.json', 'utf8');
      tsconfig = JSON.parse(tc);
    } catch {}
    let baseUrlPath = null;
    let pathsMap = {};
    if (tsconfig && tsconfig.compilerOptions) {
      const co = tsconfig.compilerOptions;
      if (co.baseUrl) {
        try { baseUrlPath = require('path').resolve(process.cwd(), co.baseUrl); } catch {}
      }
      if (co.paths && typeof co.paths === 'object') {
        pathsMap = co.paths;
      }
    }

    const resolveByTsconfig = (spec, fromFile) => {
      const path = require('path');
      if (spec.startsWith('.') || spec.startsWith('/')) {
        // Relative handled by ts-morph normally
        return null;
      }
      // paths mapping (multi-glob)
      const candidates = [];
      for (const pat in (pathsMap || {})) {
        const arr = Array.isArray(pathsMap[pat]) ? pathsMap[pat] : [];
        if (pat.includes('*')) {
          const [pre, suf] = pat.split('*');
          if (spec.startsWith(pre) && spec.endsWith(suf || '')) {
            const mid = spec.slice(pre.length, spec.length - (suf ? suf.length : 0));
            for (const tgt of arr) {
              let repl = tgt.includes('*') ? tgt.replace('*', mid) : tgt;
              const base = baseUrlPath || process.cwd();
              candidates.push(path.resolve(base, repl));
            }
          }
        } else if (spec === pat) {
          for (const tgt of arr) {
            const base = baseUrlPath || process.cwd();
            candidates.push(path.resolve(base, tgt));
          }
        }
      }
      if (!candidates.length && baseUrlPath) {
        candidates.push(path.resolve(baseUrlPath, spec));
      }
      const tryWithExt = (p) => {
        const exts = ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'];
        for (const ext of exts) {
          const f = project.getSourceFile(p + ext);
          if (f) return f;
        }
        for (const ext of exts) {
          const f = project.getSourceFile(path.join(p, 'index' + ext));
          if (f) return f;
        }
        const f2 = project.getSourceFile(p);
        return f2 || null;
      };
      for (const c of candidates) {
        const f = tryWithExt(c);
        if (f) return f;
      }
      return null;
    };

    const resolveImportSourceFile = (decl) => {
      try {
        const sf = decl.getModuleSpecifierSourceFile && decl.getModuleSpecifierSourceFile();
        if (sf) return sf;
        const mod = decl.getModuleSpecifierValue ? decl.getModuleSpecifierValue() : null;
        if (!mod) return null;
        return resolveByTsconfig(mod, decl.getSourceFile());
      } catch { return null; }
    };
    const findExportDecl = (from) => sf.getExportDeclarations().find(d => (d.getModuleSpecifierValue && d.getModuleSpecifierValue() === from));

    MESSAGES = [];
    for (let idx = 0; idx < actions.length; idx++) {
      const a = actions[idx];
      if (!a || typeof a !== 'object') continue;
      const kind = (a.kind || a.action || '').trim();
      const payload = a.payload || {};
      if (!kind) continue;
      try {
        if (kind === 'insert_import') {
          const mod = String(payload.module || payload.from || '').trim();
          if (!mod) continue;
          let decl = getImportFrom(mod);
          if (!decl) {
            const side = !!payload.namespace === false && !payload.default && !(payload.named && payload.named.length);
            decl = insertImportWithCluster({ moduleSpecifier: mod }, side);
          }
          if (payload.isType) {
            try { decl.setIsTypeOnly(true); } catch {}
          }
          // namespace import
          if (payload.namespace) {
            if (!decl.getNamespaceImport()) {
              decl.setNamespaceImport(String(payload.namespace));
            }
          } else {
            const specs = decl.getNamedImports().map(s => (s.getAliasNode() ? s.getName() + ' as ' + s.getAliasNode().getText() : s.getName()));
            const named = Array.isArray(payload.named) ? payload.named.map(v => (typeof v === 'string' ? {name: v} : v)) : (payload.named ? [{name: String(payload.named)}] : []);
            const addList = [];
            for (const n of named) {
              if (!n || !n.name) continue;
              const key = n.alias ? `${n.name} as ${n.alias}` : n.name;
              if (!specs.includes(key)) addList.push(n);
            }
            if (addList.length) {
              if (hasCommentsInNamed(decl)) {
                addNamedPreserveComments(decl, addList);
              } else {
                for (const n of addList) decl.addNamedImport(n.alias ? { name: n.name, alias: n.alias } : { name: n.name });
              }
            }
            const dflt = payload.default ? String(payload.default) : null;
            if (dflt) decl.setDefaultImport(dflt);
            // removals (optional)
            if (Array.isArray(payload.removeNamed)) {
              for (const r of payload.removeNamed) {
                const nm = String(r || '').trim();
                if (!nm) continue;
                const ni = decl.getNamedImports().find(s => s.getName() === nm || (s.getAliasNode() && (s.getName() + ' as ' + s.getAliasNode().getText()) === nm));
                if (ni) ni.remove();
              }
            }
            if (payload.removeDefault) { try { decl.removeDefaultImport(); } catch {} }
            if (payload.removeNamespace) { try { decl.removeNamespaceImport(); } catch {} }
          }
          const conflict = ensureDefaultConflictFree(mod);
          if (conflict) return fail(conflict, { action: kind, index: idx, file: sf ? sf.getFilePath() : undefined });
        } else if (kind === 'update_import') {
          const mod = String(payload.module || payload.from || '').trim();
          if (!mod) continue;
          let decl = getImportFrom(mod);
          if (!decl) decl = insertImportWithCluster({ moduleSpecifier: mod }, !!payload.namespace === false && !payload.default && !(payload.named && payload.named.length));
          if (payload.isType) {
            try { decl.setIsTypeOnly(true); } catch {}
          }
          // namespace has precedence; clears others
          if (payload.namespace) {
            decl.setNamespaceImport(String(payload.namespace));
            decl.removeDefaultImport();
            decl.removeNamedImports();
          } else {
            // default import
            if (payload.default) decl.setDefaultImport(String(payload.default));
            // named imports (allow alias objects)
            const existing = new Set(decl.getNamedImports().map(s => (s.getAliasNode() ? s.getName() + ' as ' + s.getAliasNode().getText() : s.getName())));
            const arr = Array.isArray(payload.named) ? payload.named : [];
            const addList2 = [];
            for (const spec of arr) {
              if (!spec) continue;
              if (typeof spec === 'string') {
                if (!existing.has(spec)) addList2.push({ name: spec });
              } else {
                const name = String(spec.name || '').trim();
                const alias = spec.alias ? String(spec.alias) : undefined;
                const key = alias ? `${name} as ${alias}` : name;
                if (name && !existing.has(key)) addList2.push(alias ? { name, alias } : { name });
              }
            }
            if (addList2.length) {
              if (hasCommentsInNamed(decl)) {
                addNamedPreserveComments(decl, addList2);
              } else {
                for (const s of addList2) decl.addNamedImport(s.alias ? { name: s.name, alias: s.alias } : { name: s.name });
              }
            }
            // removals
            if (Array.isArray(payload.removeNamed)) {
              for (const r of payload.removeNamed) {
                const nm = String(r || '').trim();
                if (!nm) continue;
                const ni = decl.getNamedImports().find(s => s.getName() === nm || (s.getAliasNode() && (s.getName() + ' as ' + s.getAliasNode().getText()) === nm));
                if (ni) ni.remove();
              }
            }
            if (payload.removeDefault) { try { decl.removeDefaultImport(); } catch {} }
            if (payload.removeNamespace) { try { decl.removeNamespaceImport(); } catch {} }
          }
          const conflict = ensureDefaultConflictFree(mod);
          if (conflict) return fail(conflict, { action: kind, index: idx, file: sf ? sf.getFilePath() : undefined });
        } else if (kind === 'insert_function') {
          const name = String(payload.name || payload.symbol || payload.function || '').trim();
          if (!name) continue;
          if (!sf.getFunction(name)) {
            const doc = String(payload.doc || '').trim();
            if (doc) sf.addStatements([`/** ${doc} */`]);
            sf.addStatements([`export function ${name}() {\n  // TODO\n}`]);
          }
        } else if (kind === 'insert_class') {
          const name = String(payload.name || payload.symbol || payload.class || '').trim();
          if (!name) continue;
          if (!sf.getClass(name)) {
            const doc = String(payload.doc || '').trim();
            if (doc) sf.addStatements([`/** ${doc} */`]);
            sf.addStatements([`export class ${name} {\n  constructor() {}\n}`]);
          }
        } else if (kind === 'insert_interface') {
          const name = String(payload.name || '').trim();
          if (!name) continue;
          if (!sf.getInterface(name)) {
            const doc = String(payload.doc || '').trim();
            if (doc) sf.addStatements([`/** ${doc} */`]);
            const intf = sf.addInterface({ name, isExported: true });
            const members = Array.isArray(payload.members) ? payload.members.map(String) : [];
            for (const m of members) {
              try { intf.addProperty({ name: m }); } catch {}
            }
          }
        } else if (kind === 'insert_type_alias') {
          const name = String(payload.name || '').trim();
          if (!name) continue;
          if (!sf.getTypeAlias(name)) {
            const doc = String(payload.doc || '').trim();
            if (doc) sf.addStatements([`/** ${doc} */`]);
            sf.addTypeAlias({ name, isExported: true, type: String(payload.type || 'any') });
          }
        } else if (kind === 'export_default') {
          const expr = String(payload.expression || payload.name || '').trim();
          if (expr) sf.addExportAssignment({ expression: expr, isExportEquals: false });
        } else if (kind === 'export_named') {
          const from = payload.from ? String(payload.from) : undefined;
          const arr = Array.isArray(payload.named) ? payload.named : [];
          const namedExports = [];
          for (const spec of arr) {
            if (!spec) continue;
            if (typeof spec === 'string') namedExports.push(spec);
            else {
              const name = String(spec.name || '').trim();
              const alias = spec.alias ? String(spec.alias) : undefined;
              if (name) namedExports.push(alias ? { name, alias } : name);
            }
          }
          if (namedExports.length > 0) {
            if (from) {
              let ex = findExportDecl(from);
              if (!ex) ex = sf.addExportDeclaration({ moduleSpecifier: from });
              const exist = new Set((ex.getNamedExports() || []).map(e => (e.getAliasNode() ? e.getName() + ' as ' + e.getAliasNode().getText() : e.getName())));
              for (const ne of namedExports) {
                if (typeof ne === 'string') {
                  if (!exist.has(ne)) ex.addNamedExport(ne);
                } else {
                  const key = ne.alias ? `${ne.name} as ${ne.alias}` : ne.name;
                  if (!exist.has(key)) ex.addNamedExport(ne);
                }
              }
            } else {
              // export { ... } (no from)
              // Merge with existing no-from export if present
              let ex = sf.getExportDeclarations().find(d => !d.getModuleSpecifierValue());
              if (!ex) ex = sf.addExportDeclaration({});
              const exist = new Set((ex.getNamedExports() || []).map(e => (e.getAliasNode() ? e.getName() + ' as ' + e.getAliasNode().getText() : e.getName())));
              for (const ne of namedExports) {
                if (typeof ne === 'string') {
                  if (!exist.has(ne)) ex.addNamedExport(ne);
                } else {
                  const key = ne.alias ? `${ne.name} as ${ne.alias}` : ne.name;
                  if (!exist.has(key)) ex.addNamedExport(ne);
                }
              }
            }
          }
        } else if (kind === 'rename_symbol') {
          const from = String(payload.from || payload.old || payload.symbol || '').trim();
          const to = String(payload.to || payload.new || '').trim();
          if (!from || !to || from === to) continue;
          let declSourceFile = null;
          const tryRename = () => {
            const renameDecl = (decl) => { try { declSourceFile = decl.getSourceFile(); decl.rename(to); return true; } catch { return false; } };
            // Prefer project-wide declaration search
            for (const file of project.getSourceFiles()) {
              const v = file.getVariableDeclaration(from); if (v && renameDecl(v)) return true;
              const f = file.getFunction(from); if (f && renameDecl(f)) return true;
              const c = file.getClass(from); if (c && renameDecl(c)) return true;
              const i = file.getInterface(from); if (i && renameDecl(i)) return true;
              const t = file.getTypeAlias(from); if (t && renameDecl(t)) return true;
            }
            return false;
          };
          const did = tryRename();
          if (!did) {
            // Fallback: rename matching identifiers (same semantics as textual, but skip comments/strings)
            for (const file of project.getSourceFiles()) {
              const idents = file.getDescendantsOfKind(SyntaxKind.Identifier).filter(id => id.getText() === from);
              for (const id of idents) { try { id.replaceWithText(to); } catch {} }
            }
          }
          // Optional JSX props touch
          if (payload.touchJSXProps || payload.touchJsxProps) {
            for (const file of project.getSourceFiles()) {
              const attrs = file.getDescendantsOfKind(SyntaxKind.JsxAttribute).filter(a => a.getName && a.getName() === from);
              for (const a of attrs) { try { const n = a.getNameNode(); if (n) n.replaceWithText(to); } catch {} }
            }
          }
          // Handle export kind flips on importers and re-exports (multi-hop)
          const flip = String(payload.flip || '').trim();
          if (flip && declSourceFile) {
            const srcFile = declSourceFile;
            // Build provider set via re-export traversal
            const path = require('path');
            const provider = new Set([srcFile.getFilePath()]);
            let frontier = new Set([srcFile.getFilePath()]);
            const edges = [];
            for (const file of project.getSourceFiles()) {
              for (const ex of file.getExportDeclarations()) {
                const msf = ex.getModuleSpecifierSourceFile && ex.getModuleSpecifierSourceFile();
                if (!msf) continue;
                const names = ex.getNamedExports().map(ne => ne.getName());
                const star = ex.isNamespaceExport ? true : (!names || names.length === 0);
                edges.push({ from: msf.getFilePath(), to: file.getFilePath(), names, star });
              }
            }
            const symbolName = from; // before rename
            let depthGuard = 0;
            while (frontier.size && depthGuard++ < 50) {
              const next = new Set();
              for (const f of frontier) {
                for (const e of edges) {
                  if (e.from !== f) continue;
                  // propagate if star or symbol explicitly re-exported
                  if (e.star || (e.names || []).includes(symbolName)) {
                    if (!provider.has(e.to)) { provider.add(e.to); next.add(e.to); }
                  }
                }
              }
              frontier = next;
            }
            // Update importers of any provider file
            for (const file of project.getSourceFiles()) {
              for (const d of file.getImportDeclarations()) {
                let msf = d.getModuleSpecifierSourceFile && d.getModuleSpecifierSourceFile();
                if (!msf) msf = resolveImportSourceFile(d);
                if (!msf || !provider.has(msf.getFilePath())) continue;
                if (flip === 'default_to_named') {
                  const def = d.getDefaultImport && d.getDefaultImport();
                  if (def) {
                    const local = def.getText();
                    // add named import with alias if needed
                    const exists = d.getNamedImports().some(s => s.getName() === to || (s.getAliasNode() && s.getAliasNode().getText() === local));
                    if (!exists) {
                      if (local && local !== to) d.addNamedImport({ name: to, alias: local });
                      else d.addNamedImport({ name: to });
                    }
                    try { d.removeDefaultImport(); } catch {}
                  }
                } else if (flip === 'named_to_default') {
                  // find named matching 'from' or alias
                  const nm = d.getNamedImports().find(s => s.getName() === from || (s.getAliasNode() && s.getAliasNode().getText() === from));
                  if (nm) {
                    const aliasNode = nm.getAliasNode();
                    const local = aliasNode ? aliasNode.getText() : from;
                    try { d.setDefaultImport(local); } catch {}
                    try { nm.remove(); } catch {}
                  }
                }
              }
            }
            // Update re-exports with named specs
            for (const file of project.getSourceFiles()) {
              for (const ex of file.getExportDeclarations()) {
                let msf = ex.getModuleSpecifierSourceFile && ex.getModuleSpecifierSourceFile();
                if (!msf) {
                  try { msf = resolveByTsconfig(ex.getModuleSpecifierValue(), file); } catch {}
                }
                if (!msf || !provider.has(msf.getFilePath())) continue;
                const specs = ex.getNamedExports();
                if (!specs || specs.length === 0) continue;
                for (const s of specs) {
                  const name = s.getName();
                  const alias = s.getAliasNode() ? s.getAliasNode().getText() : undefined;
                  if (flip === 'default_to_named' && name === 'default') {
                    try { s.replaceWithText(alias ? `${to} as ${alias}` : to); } catch {}
                  } else if (flip === 'named_to_default' && (name === from || alias === from)) {
                    const al = alias || name;
                    try { s.replaceWithText(`default as ${al}`); } catch {}
                  }
                }
              }
            }
          }
        }
      } catch (e) {
        MESSAGES.push({ level: 'error', action: kind, index: idx, file: sf ? sf.getFilePath() : undefined, reason: (e && e.message) ? e.message : String(e) });
      }
    }
    // Optional Prettier formatting if available
    const prettierOptOut = (process.env.NERION_PRETTIER || '').trim().toLowerCase() in { '0':true, 'false':true, 'no':true };
    let prettier = null;
    if (!prettierOptOut) {
      try { prettier = require('prettier'); } catch {}
    }

    const formatIf = (text, filePath) => {
      if (!prettier) return text;
      try {
        // Honor .prettierignore
        try {
          if (prettier.getFileInfo) {
            const info = prettier.getFileInfo.sync ? prettier.getFileInfo.sync(filePath, { ignorePath: '.prettierignore' }) : null;
            if (info && info.ignored) return text;
          }
        } catch {}
        return prettier.format(text, { filepath: filePath, singleQuote: true });
      } catch { return text; }
    };

    if (files) {
      const outFiles = project.getSourceFiles().map(f => ({ path: f.getFilePath(), source: formatIf(f.getFullText(), f.getFilePath()) }));
      const resp = { ok: true, files: outFiles };
      if (Array.isArray(MESSAGES) && MESSAGES.length) resp.messages = MESSAGES;
      process.stdout.write(JSON.stringify(resp));
    } else {
      const out = formatIf(sf.getFullText(), 'file.tsx');
      const resp = { ok: true, source: out };
      if (Array.isArray(MESSAGES) && MESSAGES.length) resp.messages = MESSAGES;
      process.stdout.write(JSON.stringify(resp));
    }
  } catch (e) {
    return fail(e && e.message ? e.message : 'unknown');
  }
}

main();
